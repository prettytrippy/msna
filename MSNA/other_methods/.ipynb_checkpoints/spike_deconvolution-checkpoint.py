from msna_cnn import MSNA_CNN
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
from scipy.signal import butter, filtfilt, medfilt, deconvolve, find_peaks
from scipy.optimize import minimize, differential_evolution

class SpikeDeconvolverPipeline():
    def __init__(self, verbose=False, n=256, sr=250):
        self.sr = sr
        self.n = n
        self.verbose = verbose
        self.init_params()

    def init_params(self):
        n = self.n
        self.kernel = np.zeros(n)
        self.kernel[n//2] = 1.0
        
        
        self.distance=self.sr
        self.height=0.0 
        self.threshold=0.0
        self.prominence=0.0

    def save(self, output_folder):
        json_dict = {
            "height":self.height,
            "threshold":self.threshold,
            "prominence":self.prominence,
            "kernel":self.kernel.tolist(),
        }
        with open(f"{output_folder}/params.json", 'w') as file:
            json.dump(json_dict, file)

    def load(self, output_folder):
        with open(f"{output_folder}/params.json", 'r') as file:
            json_dict = json.load(file)
        self.height = json_dict['height']
        self.threshold = json_dict['threshold']
        self.prominence = json_dict['prominence']
        self.kernel = np.array(json_dict['kernel'])

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

    def mad_noise_estimate(self, signal):
        median = np.median(signal)
        mad = 1.4826 * np.median(np.abs(signal - median))
        return mad**2  # Noise power

    def wiener_deconvolution(self, signal, psf):
        noise_power = self.mad_noise_estimate(signal)
        # FFT of signal and point spread function (PSF)
        signal_fft = np.fft.fft(signal)
        psf_fft = np.fft.fft(psf, n=len(signal))
        
        # Wiener filter in frequency domain
        psf_fft_conj = np.conj(psf_fft)
        filter_fft = psf_fft_conj / (psf_fft * psf_fft_conj + noise_power)
        result_fft = signal_fft * filter_fft
        
        # Inverse FFT to get the deconvolved signal
        return np.fft.ifft(result_fft).real
    
    def deconvolve(self, df):
        signal = df['normalized MSNA'].to_numpy()
        # n = len(signal)
        # fft_signal = np.fft.fft(signal, n=n)
        # fft_kernel = np.fft.fft(self.kernel, n=n)
        # fft_spikes = fft_signal / (fft_kernel + 1e-8)
        # deconvolved = np.abs(np.fft.ifft(fft_spikes, n=n))
        # return deconvolved
        return self.wiener_deconvolution(signal, self.kernel)
        # return deconvolve(signal, self.kernel)[0]

    def predict_labels(self, df):
        deconvolved_spikes = self.deconvolve(df)
        indices = find_peaks(deconvolved_spikes, 
                          height=self.height, 
                          threshold=self.threshold, 
                          prominence=self.prominence,
                          distance=self.distance
                         )[0]
        return indices

    def predict_probabilities(self, df):
        deconvolved_spikes = self.deconvolve(df)
        return deconvolved_spikes

    def threshold_probabilities(self, coeffs, placeholder=None):
        peaks = find_peaks(coeffs, 
                          height=self.height, 
                          threshold=self.threshold, 
                          prominence=self.prominence,
                          distance=self.distance
                         )[0]
        return peaks
        
        thresholded_probabilities = np.array(coeffs > self.threshold, dtype=int)
        labels = np.nonzero(thresholded_probabilities)[0]
        return labels

    def predict_labels(self, df):
        coeffs = self.predict_probabilities(df)
        return self.threshold_probabilities(coeffs, None)

    def get_kernel(self, df):
        actual_bursts = self.get_burst_idxs(df)
        signal = df['normalized MSNA'].to_numpy()
        n = len(self.kernel)
        actual_bursts = actual_bursts[(actual_bursts > n) & (actual_bursts < len(signal) - n)]
        return np.mean([signal[idx - n//2: idx + n//2] for idx in actual_bursts], axis=0)

    def train(self, filenames, maxiter=256):
        if self.verbose:
            print("Processing dataframes.")

        if type(filenames) != list:
            filenames = [filenames]
            
        dfs = [self.process_file(filename) for filename in filenames]
        dfs = [df for df in dfs if df is not None]

        if self.verbose:
            print("Processed dataframes, finding kernel.")

        self.kernel = np.mean([self.get_kernel(df) for df in dfs], axis=0)

        zipped_probabilities = [(self.predict_probabilities(df), df) for df in dfs]

        def loss(x):
            self.height, self.threshold, self.prominence = x
            total = 0
            count = 0
            for coeffs, df in zipped_probabilities:
                n = training_metrics(self, df, coeffs, None)[2]
                if n > 0.0:
                    total += 1 - n
                    count += 1
                if count < 1:
                    return float('inf')
            return total/count

        result = differential_evolution(loss, bounds=[(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)])
        self.height, self.threshold, self.prominence = result.x

        if self.verbose:
            print(f"Kernel found")
                
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