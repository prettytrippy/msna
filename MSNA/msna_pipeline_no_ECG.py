from msna_cnn_no_ECG import MSNA_CNN
from msna_common import read_msna, metrics, training_metrics
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import glob
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt, find_peaks
from scipy.optimize import minimize, differential_evolution
from scipy.ndimage import maximum_filter

# TODO: Add downsampling to get signals in a fixed sampling rate

class MSNA_pipeline():
    
    def __init__(self, sampling_rate=250, window_size=256, batch_size=1024, verbose=True, fc_dim=32, dropout_prob=0.5, 
                 argmax_prop=0.1, training_imbalance=0.5, rectify_prop=0.1, high_freq=0.25, low_freq=30.0):
        self.verbose = verbose
        self.sr = sampling_rate
        self.window_size = window_size
        self.batch_size = batch_size
        self.fc_dim=fc_dim
        self.dropout_prob=dropout_prob
        self.argmax_window=int(argmax_prop * self.window_size)
        self.training_imbalance = training_imbalance
        self.rectify_prop = rectify_prop
        self.low_freq = low_freq
        self.high_freq = high_freq

        self.init_params()

    def init_params(self):
        self.cnn = MSNA_CNN(n=self.window_size, dropout_prob=self.dropout_prob, fc_dim=self.fc_dim)
        self.threshold = 0.90

    def save(self, output_folder):
        torch.save(self.cnn.state_dict(), f"{output_folder}/model_parameters.pth")
        with open(f"{output_folder}/pipeline_parameters.json", 'w') as file:
            json_dict = {"threshold":self.threshold}
            file.write(json.dumps(json_dict))

    def load(self, output_folder):
        self.cnn.load_state_dict(torch.load(f"{output_folder}/model_parameters.pth"))
        with open(f"{output_folder}/pipeline_parameters.json", 'r') as file:
            json_dict = json.loads(file.read())
            self.threshold = json_dict['threshold']

    def get_burst_idxs(self, df):
        return np.nonzero(df['BURST'].to_numpy())[0]

    def rectify_bursts(self, df):
        ws = int(self.sr * self.rectify_prop)
        
        msna = df['normalized MSNA'].to_numpy()
        indices = self.get_burst_idxs(df)
        
        new_indices = []
        for idx in indices:
            start, end = max(0, idx-ws//2), min(idx+ws//2, len(msna))
            max_idx = np.argmax(msna[start:end]) + start
            new_indices.append(max_idx)

        rectified_bursts = np.zeros(len(msna))
        rectified_bursts[new_indices] = 1
        df['rectified BURST'] = rectified_bursts
    
    def process_dataframe(self, df):
        low_cutoff = self.low_freq   # Low cutoff hz
        high_cutoff = self.high_freq  # High cutoff hz
        fs = self.sr 
        
        df['Integrated MSNA'] = medfilt(df['Integrated MSNA'], 3)
        df['ECG'] = medfilt(df['ECG'], 3)
        
        df['band-pass MSNA'] = self.band_pass_filter(df['Integrated MSNA'], low_cutoff, high_cutoff, fs)
        # df['band-pass ECG'] = self.band_pass_filter(df['ECG'], low_cutoff, high_cutoff, fs)

        df['normalized MSNA'] = self.median_normalize(df['band-pass MSNA'])
        # df['normalized ECG'] = self.median_normalize(df['band-pass ECG'])
        # df['normalized MSNA'] = self.median_normalize(df['Integrated MSNA'])
        # df['normalized ECG'] = self.median_normalize(df['ECG'])

        self.rectify_bursts(df)

        return df

    def process_file(self, filename):
        df = read_msna(filename)
        if df is not None:
            return self.process_dataframe(df)
        return None
    
    def band_pass_filter(self, data, low_cutoff, high_cutoff, fs, order=3):
        nyquist = 0.5 * fs
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = butter(order, [low,high], btype='band', analog=False)
        y = filtfilt(b, a, data)
        return y

    def median_normalize(self, data):
        numpy_data = data.to_numpy()  
        mu = np.median(numpy_data, axis=0)
        difference = np.abs(numpy_data - mu)
        sigma = np.median(difference, axis=0)
        return (numpy_data - mu) / sigma
    
    def chunk_df1(self, df):
        n = self.window_size
        
        msna = df['normalized MSNA'].to_numpy()
        # ecg = df['normalized ECG'].to_numpy()
        burst_labels = df['BURST'].to_numpy()

        indices = self.find_peaks(msna)
        chunks = [
                [torch.tensor(msna[idx-n//2: idx+n//2]), 
                int(1 in burst_labels[idx-n//2: idx+n//2])]
            for idx in indices
        ]

        return chunks

    def chunk_df(self, df):
        msna = df['normalized MSNA'].to_numpy()
        burst = df['rectified BURST'].to_numpy()
        length = len(df)
        half_w = self.window_size // 2
        noise_std = 0.01
    
        # 1) Collect burst indices and optionally merge or skip duplicates
        burst_idxs = np.where(burst > 0)[0]
        # (Optional) burst_idxs = merge_close_bursts(burst_idxs, threshold=some_value)
    
        # 2) Positive windows
        pos_chunks = []
        for center in burst_idxs:
            # Jitter
            start = center - half_w
            end   = center + half_w
            # Bounds check
            if start < 0 or end > length:
                continue
            # Slice
            msna_chunk = msna[start:end]
    
            x = torch.tensor(msna_chunk)
            pos_chunks.append((x, 1))

        # 3) Negative windows

        peak_indices = self.find_peaks(msna)
        
        neg_chunks = []
        n_neg = int(len(pos_chunks) * self.training_imbalance + 1)
        attempts = 0
        attempt_limit = len(peak_indices)
        while len(neg_chunks) < n_neg and attempts < attempt_limit:
            attempts += 1
            # center = np.random.randint(half_w, length - half_w)
            center = np.random.choice(peak_indices)
            peak_indices = peak_indices[peak_indices != center]
            
            # Random exclusion zone
            # exclusion_factor = np.random.uniform(exclusion_min, exclusion_max)
            # # Ensure center is not too close to any burst
            # if np.any(np.abs(center - burst_idxs) < (exclusion_factor * self.window_size)):
            #     continue
    
            start = center - half_w
            end   = center + half_w

            if 1 in burst[start:end]:
                continue
                
            msna_chunk = msna[start:end]
    
            # (Optional) Data augmentation: noise, scaling, etc.
    
            x = torch.tensor(msna_chunk)
            neg_chunks.append((x, 0))
    
        # 4) Combine, shuffle, return
        dataset = pos_chunks + neg_chunks
        np.random.shuffle(dataset)
        return dataset
        
    def make_training_set(self, chunks):
        trainloader = torch.utils.data.DataLoader(chunks, shuffle=True, batch_size=self.batch_size)
        return trainloader

    def train_cnn(self, trainloader, num_epochs=75, learning_rate=0.001):
        self.cnn.train()
        
        # Loss function, optimizer, scheduler
        criterion = nn.BCELoss() 
        optimizer = optim.Adam(self.cnn.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=self.verbose)
        
        # Training loop
        for epoch in range(num_epochs):
            running_loss = 0.0
        
            for inputs, labels in trainloader:
                # Forward
                optimizer.zero_grad()
                outputs = self.cnn(inputs.float().unsqueeze(1))
                # outputs = self.cnn(inputs.float().squeeze().unsqueeze(1))
                
                # Backward
                loss = criterion(outputs, labels.float().unsqueeze(1))
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
        
            # Average loss
            avg_loss = running_loss / len(trainloader)
            
            if self.verbose:
                if (epoch+1) % 8 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            
            scheduler.step(avg_loss)
        
        if self.verbose:
            print("Training complete.")
            
        self.cnn.eval()

    def argmax_filter(self, arr, window_size):
        if arr.size < window_size:
            return []
        windows = sliding_window_view(arr, window_size)
        relative_indices = np.argmax(windows, axis=1)
        absolute_indices = np.arange(len(relative_indices)) + relative_indices
        return absolute_indices

    def find_peaks(self, signal):
        # Derivative method
        # signal = np.convolve(signal, [0.1, 0.2, 0.4, 0.2, 0.1], mode='same')
        # diff = np.diff(signal)
        
        # increasing = diff > 0
        # decreasing = diff < 0
        
        # idxs = np.where(increasing[:-1] & decreasing[1:])[0] + 1

        # All indices method
        # idxs = np.arange(0, len(signal), 4)

        # Argmax method
        idxs = list(set(self.argmax_filter(signal, self.argmax_window)))
        idxs.sort()
        idxs = np.array(idxs)

        n = self.window_size
        idxs = idxs[(idxs > n//2) & (idxs < (len(signal) - n//2))]
        return idxs
        
    def make_sample_list(self, df, train=False):
        chunks = []
        n = self.window_size
        
        msna = df['normalized MSNA'].to_numpy()
        
        idxs = self.find_peaks(msna)
        
        for idx in idxs:
            start = idx - n//2
            end = idx + n//2
        
            chunk = msna[start:end]
            
            chunks.append(chunk)
    
        return np.array(chunks), idxs

    def clean_predicted_peaks(self, indices, tolerance=25):
        if indices.size < 1:
            return indices
            
        indices = np.sort(indices) 
        cleaned_indices = []
    
        current_group = [indices[0]]
    
        for idx in indices[1:]:
            if idx - current_group[-1] <= tolerance:
                current_group.append(idx)  
            else:
                cleaned_indices.append(int(np.mean(current_group)))
                current_group = [idx] 
                
        cleaned_indices.append(int(np.mean(current_group)))
        return np.array(cleaned_indices)
        
    def predict_probabilities(self, df):
        # Get all chunks (with a possible peak index in the center)
        chunks, possible_peak_indices = self.make_sample_list(df)
    
        # Run the model to get output probabilities
        model_input = torch.tensor(chunks)
        with torch.no_grad():
            probabilities = self.cnn(model_input.float().unsqueeze(1)).detach()
        probabilities = probabilities.numpy().squeeze()

        return probabilities, possible_peak_indices

    def threshold_probabilities(self, probabilities, possible_peak_indices):
        thresholded_probabilities = np.array(probabilities > self.threshold, dtype=int)
        labels = np.nonzero(thresholded_probabilities)[0]
        
        # Get the indices of predicted thresholded peaks
        peak_indices = possible_peak_indices[labels]

        peak_indices = self.clean_predicted_peaks(peak_indices)
    
        return peak_indices

    def predict_labels(self, df):
        probabilities, possible_peak_indices = self.predict_probabilities(df)
        return self.threshold_probabilities(probabilities, possible_peak_indices)

    def train_threshold(self, dfs, maxiter=16):
        # Preprocess model outputs so we can quickly find threshold
        # This code looks awful, but it makes threshold optimization fast
        
        zipped_probabilities = [(self.predict_probabilities(df), df) for df in dfs]

        def loss(x):
            self.threshold = x
            total = 0
            count = 0
            for (probs, indices), df in zipped_probabilities:
                n = training_metrics(self, df, probs, indices)[2]
                if n > 0.0:
                    total += 1 - n
                    count += 1
                if count < 1:
                    return float('inf')
            return total/count

        result = differential_evolution(loss, x0=self.threshold, bounds=[(0.0, 1.0 - 1e-10)], maxiter=maxiter)

        self.threshold = result.x[0]
        
        if self.verbose:
            print("Threshold result:", result)
            
    def train(self, filenames, threshold_train_max_iter=32, learning_rate=0.001, num_epochs=32):
        if self.verbose:
            print("Processing dataframes.")

        if type(filenames) != list:
            filenames = [filenames]
            
        dfs = [self.process_file(filename) for filename in filenames]
        dfs = [df for df in dfs if df is not None]


        if self.verbose:
            print("Processed dataframes, chunking.")
        
        chunks = []
        for df in dfs:
            chunk = self.chunk_df(df)
            chunks.extend(chunk)
            
        if self.verbose:
            print("Got chunks, making dataloader.")

        trainloader = self.make_training_set(chunks)

        if self.verbose:
            print("Made dataloader, training CNN.")

        self.train_cnn(trainloader, num_epochs=num_epochs, learning_rate=learning_rate)
        # self.find_kernel(dfs)

        if self.verbose:
            print("trained CNN, getting threshold.")

        self.train_threshold(dfs, maxiter=threshold_train_max_iter)

        if self.verbose:
            print("Got threshold, training complete.")

    def predict(self, filename):
        df = self.process_file(filename)
        if df is None:
            raise ValueError(f"Could not read MSNA file {filename}")
        return self.predict_labels(df)

    def split_data(self, data, k):
        length = len(data) // k
        splits = []
        for i in range(k-1):
            chunk = data[i*length:(i+1)*length]
            if type(data) != list: # Handle intrasubject Dataframe chunks
                chunk = [chunk]
            splits.append(chunk)
        return splits
        
    def k_fold_cross_validation(self, filenames, k):
        k += 1
        splits = self.split_data(filenames, k)

        n = len(splits)
        
        f1s = []
        precisions = []
        recalls = []

        for idx in range(n):
            
            train_samples = []
            for jidx in range(n):
                if jidx != idx:
                    train_samples.extend(splits[jidx])
                    
            test_sample = splits[idx]

            self.init_params()

            if self.verbose:
                print("Training")
            self.train(train_samples)

            if self.verbose:
                print("Validating")
                
            test_sample = [self.process_file(filename) for filename in test_sample]
            test_sample = [df for df in test_sample if df is not None]
            precision, recall, f1 = 0, 0, 0
            for i in test_sample:
                new_precision, new_recall, new_f1 = metrics(self, i, verbose=False)
                precision += new_precision
                recall += new_recall
                f1 += new_f1
                
            f1 /= len(test_sample)
            recall /= len(test_sample)
            precision /= len(test_sample) 
            
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)

            if self.verbose:
                print(f"F1 for split {idx+1}: {f1}")

            # if self.verbose:
            #     print("Current Threshold:", self.threshold)
            # Add info about this run (threshold, etc.)

        return precisions, recalls, f1s