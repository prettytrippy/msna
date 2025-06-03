import glob
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import rankdata
import pandas as pd
from tqdm import tqdm
from typing import Tuple

def msna_metric(
    pred_peaks: np.ndarray, true_peaks: np.ndarray, window_size: int = 40
) -> Tuple[float, float, float]:
    """
    An F1 metric based on a custom ground-truth binning scheme for this task. 
    
    Args:
        pred_peaks (np.ndarray): An array of integers representing where each 
            MSNA peak was found.
        true_peaks (np.ndarray): An array of integers representing where each 
            ground truth MSNA peak is. This can be computed as `peaks_from_bool_1d(df["Burst"])`
        window_size (int): The size of the window around each true peak.
    
    Returns:
        float: The F1 score. 
    """
    pred, true = bin_predictions(pred_peaks, true_peaks, window_size)
    return scores(pred, true)

def bin_predictions(
    pred_peaks: np.ndarray, true_peaks: np.ndarray, window_size: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom binning scheme for the predicted and ground truth data. For each
    ground truth peak, we take some window around it and consider that as the
    true range. Any time-point that is not apart of this window is considered a
    false section. In this sense, this metric is NOT commutative and would be
    very sensitive to swapping the pred and true peaks. 

    Args:
        pred_peaks (np.ndarray): An array of integers representing where each 
            MSNA peak was found.
        true_peaks (np.ndarray): An array of integers representing where each 
            ground truth MSNA peak is. This can be computed as `peaks_from_bool_1d(df["Burst"])`
        window_size (int): The size of the window around each true peak.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: The pred and true binned data (in this order).
    """
    l = true_peaks - window_size
    r = true_peaks + window_size

    pred = []
    true = []
    for i in range(len(l)):
        # Check if there's at least one prediction in this window
        if np.sum((pred_peaks >= l[i]) & (pred_peaks < r[i])) >= 1: 
            pred.append(True)
            true.append(True)
        else:
            pred.append(False)
            true.append(True)
            
        # Check gap to next peak (only if not the last peak)
        if i < len(l) - 1:
            if np.sum((pred_peaks >= r[i]) & (pred_peaks < l[i+1])) >= 1: 
                pred.append(True)
                true.append(False)
            else:
                pred.append(False)
                true.append(False)
    
    return np.array(pred), np.array(true)

def scores(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float, float]:
    """An F1 metric based on boolean `pred` and `true` aligned data."""
    if np.array(pred.shape) != np.array(true.shape):
        raise ValueError("`pred` and `true` arrays must have the same shape.")

    tp, fp, tn, fn = confusion_matrix_values(pred, true)
    
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1, precision, recall
 
def confusion_matrix_values(
    pred: np.ndarray, true: np.ndarray
) -> Tuple[float, float, float, float]:
    """Get the true/false positive/negatives of the boolean data."""
    tp = np.sum((pred == 1) & (true == 1))
    fp = np.sum((pred == 1) & (true == 0))
    tn = np.sum((pred == 0) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))
    
    return tp, fp, tn, fn

def peaks_from_bool_1d(bool_array: np.ndarray) -> np.ndarray: 
    """Converts a boolean array to an array of integer indices."""
    return np.where(bool_array)[0]

def get_burst_idxs(msna):
    return np.nonzero(msna['BURST'].to_numpy())[0]

def histogram_normalize(signal, a=1.0):
    ranks = rankdata(signal, method='average') / len(signal)
    scaled = np.log1p(a * ranks)
    return scaled
            
# Function to read and parse the data file
def read_msna(file_path):
    try:
        
        # Dictionary to hold metadata
        metadata = {}
        # List to hold data values
        data = []
        
        # Read file
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            lines = file.readlines()
    
        data = []
        channel_titles = []
    
        for line in lines:
            if line.startswith("ChannelTitle="):
                # Extract channel titles
                channel_titles = line.strip().split('\t')[1:]
            elif line.strip() and not line.startswith(("Interval=", "ExcelDateTime=", "TimeFormat=", "DateFormat=", "ChannelTitle=", "Range=", "UnitName=", "TopValue=", "BottomValue=")):
                # Split the line by tab
                parts = line.strip().split('\t')
                # Extract BURST comment if present
                burst_comment = ""
    
                # TODO: CHECK OTHER BURST NAMES
                if "BURST" in line:
                    burst_comment = list(parts[-1])[4:]
                    burst_comment = "".join(burst_comment)
                    burst_flag = 1
                else:
                    burst_flag = 0
                
                # Extract relevant data
                timestamp = parts[0]
                ecg = parts[1]  # ECG is the first channel after Timestamp
                nibp = parts[2]
                handgrip = parts[3]
                respiratory_waveform = parts[4]
                systolic = parts[5]
                diastolic = parts[6]
                heart_rate = parts[7]
                raw_msna = parts[8]
                stimulator = parts[9]
                filtered_msna = parts[10]
                integrated_msna = parts[11]  # Integrated MSNA is the 12th channel
                respiratory_rate = parts[12]
                filter_other = parts[13]
                percent_mvc = parts[14]
                flag = parts[15]
                channel_17 = parts[16]
                mabp = parts[17]
                
                data.append([timestamp, ecg, nibp, handgrip, respiratory_waveform, systolic, diastolic, heart_rate, raw_msna, stimulator, filtered_msna, integrated_msna, respiratory_rate, filter_other, percent_mvc, percent_mvc, flag, channel_17, mabp, burst_flag])
    
        if not channel_titles:
            print("Error: Channel titles were not found in the metadata.")
            return None
      
        # Convert to DataFrame
        # new_channels = ['Timestamp', 'ECG', 'NIBP', 'Raw MSNA', 'Integrated MSNA', 'BURST']
        new_channels = ['Timestamp', "ECG",	'NIBP', 'Handgrip', 'Respiratory Waveform', 'Systolic', 'Diastolic', 'HR', 'Raw MSNA', 'Stimulator', 'Filtered MSNA', 'Integrated MSNA', 'Respiratory Rate', 'Filter MSNA other', '%MVC', '%MVC smoothed', 'FLAG', 'Channel 17', 'Mean Arterial BP', 'BURST']

        df = pd.DataFrame(data, columns=new_channels)
        df = df.apply(pd.to_numeric, errors='coerce')
    
        for channel in new_channels:
            df[channel] = pd.to_numeric(df[channel], errors='coerce')
        if len(get_burst_idxs(df)) == 0:
            print(f"No bursts found in file {file_path}")
            return None
        
        return df
    except:
        return None

exclusions = ['205', '172']
def exclude(filename):
    for exclusion in exclusions:
        if exclusion in filename:
            return True
    return False

def get_files(glob_regex="../MSNAS/MSNA*/MSNA*burstcomments*.txt"):
    file_list = glob.glob(glob_regex) 
    file_list = [filename for filename in file_list if not exclude(filename)]
    return file_list
        
def get_dataframes(glob_regex="../MSNAS/MSNA*/MSNA*burstcomments*.txt"):
    file_list = glob.glob(glob_regex) 
    dfs = []

    file_list = [filename for filename in file_list if not exclude(filename)]
                
    for filename in tqdm(file_list):
        try:
            df = read_msna(filename)
            if 1 in df['BURST']:
                dfs.append(df)
        except Exception as e:
            print(e, filename)
    
    return dfs

def band_pass_filter(data, low_cutoff, high_cutoff, fs, order=3):
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low,high], btype='band', analog=False)
    y = filtfilt(b, a, data)
    return y

def bersen_threshold(signal):
    threshold = (np.max(signal) + np.min(signal)) / 2
    return np.array(signal > threshold, dtype=int)
    
def ecg_peaks(df, fs=250):
    ecg = df['ECG'].to_numpy()
    ecg = band_pass_filter(ecg, 3.0, 20.0, fs)
    diff = np.diff(ecg)
    square = diff**2
    smoothed = np.convolve(square, [0.1, 0.2, 0.4, 0.2, 0.1], mode='same')
    thresholded = bersen_threshold(smoothed)
    return np.nonzero(thresholded)[0]

def clean_peaks(indices, tolerance=25):
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
    
def old_stats(actual_bursts, predicted_bursts):
        # Define a tolerance for matching peaks (e.g., 50 sample indices)
        tolerance = 25
    
        # Convert the lists to numpy arrays for easier manipulation
        detected_peaks = np.array(predicted_bursts)
        actual_peaks = np.array(actual_bursts)
    
        if len(detected_peaks) == 0:
            return None
        
        # Initialize true positives (TP), false positives (FP), and false negatives (FN)
        TP = 0
        FP = 0
        FN = 0
        TN = 0  # True Negatives (not typically used in this context but included for completeness)
        
        # Calculate TP and FP
        for dp in detected_peaks:
            if np.any(np.abs(actual_peaks - dp) <= tolerance):
                TP += 1
            else:
                FP += 1
        
        # Calculate FN
        for ap in actual_peaks:
            if not np.any(np.abs(detected_peaks - ap) <= tolerance):
                FN += 1
        
        # Calculate TN
        # Note: In signal processing contexts like this, true negatives (TN) are not typically calculated 
        # because it would require accounting for all the points where no peaks are detected or expected.
        # However, for completeness, if I am considering TN as all points not being peaks, it can be approximated:
        total_samples = max(np.max(detected_peaks), np.max(actual_peaks)) + 1
        total_non_peaks = total_samples - len(detected_peaks) - len(actual_peaks)
        TN = total_non_peaks

        return TP, FP, TN, FN

def new_stats(self, df, actual_bursts, predicted_bursts):
        fs = self.sr

        ecg_indices = ecg_peaks(df, fs=fs)
        ecg_indices = clean_peaks(ecg_indices, tolerance=25)
    
        # Convert the lists to numpy arrays for easier manipulation
        detected_peaks = np.array(predicted_bursts)
        actual_peaks = np.array(actual_bursts)
    
        if len(detected_peaks) == 0:
            return None
        
        # Initialize true positives (TP), false positives (FP), and false negatives (FN)
        TP = 0
        FP = 0
        FN = 0
        TN = 0  # True Negatives (not typically used in this context but included for completeness)

        last_idx = 0
    
        for idx in ecg_indices:
            actually_in = np.any((last_idx < actual_peaks) & (actual_peaks < idx))
            guessed_in = np.any((last_idx < detected_peaks) & (detected_peaks < idx))
            
            if actually_in:
                if guessed_in:
                    TP += 1
                else:
                    FN += 1
            else:
                if guessed_in:
                    FP += 1
                else:
                    TN += 1
                    
            last_idx = idx

        return TP, FP, TN, FN

def metrics(self, df, verbose=False):
        actual_bursts = self.get_burst_idxs(df)
        predicted_bursts = self.predict_labels(df)

        f1, precision, recall = msna_metric(predicted_bursts, actual_bursts)
        
        # stats = new_stats(self, df, actual_bursts, predicted_bursts)
        # if stats == None:
        #     return 0.0, 0.0, 0.0
            
        # TP, FP, TN, FN = stats
    
        # # Calculate Precision, Recall, F1-score, and Accuracy
        # precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        # recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        # f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        # accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    
        return (precision, recall, f1)
        
        # stats = new_stats(self, df, actual_bursts, predicted_bursts)
        # if stats == None:
        #     return 0.0, 0.0, 0.0
            
        # TP, FP, TN, FN = stats
        
        # # Calculate Precision, Recall, F1-score, and Accuracy
        # precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        # recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        # f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        # accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    
        # if verbose:
        #     print(f"True Positives (TP): {TP}")
        #     print(f"False Positives (FP): {FP}")
        #     print(f"False Negatives (FN): {FN}")
        #     print(f"True Negatives (TN): {TN}")
        #     print(f"Precision: {precision:.2f}")
        #     print(f"Recall: {recall:.2f}")
        #     print(f"F1-score: {f1_score:.2f}")
        #     print(f"Accuracy: {accuracy:.2f}")
    
        # return (precision, recall, f1_score)

def training_metrics(self, df, probabilities, possible_burst_indices):

        actual_bursts = self.get_burst_idxs(df)
        predicted_bursts = self.threshold_probabilities(probabilities, possible_burst_indices)

        f1, precision, recall = msna_metric(predicted_bursts, actual_bursts)
        
        # stats = new_stats(self, df, actual_bursts, predicted_bursts)
        # if stats == None:
        #     return 0.0, 0.0, 0.0
            
        # TP, FP, TN, FN = stats
    
        # # Calculate Precision, Recall, F1-score, and Accuracy
        # precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        # recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        # f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        # accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    
        return (precision, recall, f1)
