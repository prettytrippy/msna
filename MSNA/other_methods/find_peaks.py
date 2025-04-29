from msna_common import read_msna, metrics, training_metrics
import pandas as pd
import numpy as np
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
from scipy.optimize import minimize, dual_annealing, differential_evolution

class PeakFinderPipeline():
    def __init__(self, sr=250, verbose=False):
        self.verbose = verbose
        self.sr = sr
        self.init_params()

    def init_params(self):
        self.distance=self.sr
        self.height=0.0 
        self.threshold=0.0
        self.prominence=0.0

    def save(self, output_folder):
        json_dict = {
            "height":self.height,
            "threshold":self.threshold,
            "prominence":self.prominence
        }
        with open(f"{output_folder}/params.json", 'w') as file:
            json.dump(json_dict, file)

    def load(self, output_folder):
        with open(f"{output_folder}/params.json", 'r') as file:
            json_dict = json.load(file)
        self.height=json_dict['height']
        self.threshold=json_dict['threshold']
        self.prominence=json_dict['prominence']
        
    def get_burst_idxs(self, df):
        return np.nonzero(df['BURST'].to_numpy())[0]
    
    def process_dataframe(self, df):
        low_cutoff = 0.5   # Low cutoff hz
        high_cutoff = 20  # High cutoff hz
        fs = self.sr 
        
        df['Integrated MSNA'] = medfilt(df['Integrated MSNA'], 3)
        df['ECG'] = medfilt(df['ECG'], 3)
        
        df['band-pass MSNA'] = self.band_pass_filter(df['Integrated MSNA'], low_cutoff, high_cutoff, fs)
        df['band-pass ECG'] = self.band_pass_filter(df['ECG'], low_cutoff/2, high_cutoff*2, fs)

        df['normalized MSNA'] = self.median_normalize(df['band-pass MSNA'])
        df['normalized ECG'] = self.median_normalize(df['band-pass ECG'])

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

    def clean_predicted_peaks(self, indices, tolerance=25):
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
        
    def predict_labels(self, df):
        signal = df['normalized MSNA'].to_numpy()
        idxs = find_peaks(signal, 
                          height=self.height, 
                          threshold=self.threshold, 
                          prominence=self.prominence,
                          distance=self.distance
                         )[0]
        if idxs.size > 0:
            return self.clean_predicted_peaks(idxs)
        return idxs
    
    def train(self, filenames, maxiter=256):
        if self.verbose:
            print("Processing dataframes.")

        if type(filenames) != list:
            filenames = [filenames]
            
        dfs = [self.process_file(filename) for filename in filenames]
        dfs = [df for df in dfs if df is not None]

        if self.verbose:
            print("Processed dataframes, finding parameters.")

        def loss(x):
            self.height, self.prominence, self.threshold = x
            total = 0
            count = 0
            for df in (dfs):
                n = metrics(self, df)[2]
                if n > 0.0:
                    total += 1 - n
                    count += 1
            if count < 1:
                return float('inf')
            return total/count

        initial_params = [self.height, self.prominence, self.threshold]
        # result = minimize(loss, x0=initial_params, method='Nelder-Mead', options={'disp': self.verbose, 'maxiter': maxiter})
        # result = dual_annealing(loss, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], maxiter=maxiter)
        result = differential_evolution(loss, bounds=[(-100.0, 100.0), (-100.0, 100.0), (-100.0, 100.0)])
        self.height, self.prominence, self.threshold = result.x

        if self.verbose:
            print(f"Found params.")
                
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
                new_precision, new_recall, new_f1 = metrics(self, i)
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