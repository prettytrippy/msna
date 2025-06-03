from msna.msna_common import ecg_peaks
import numpy as np

def burst_frequency(df, sr=250):
    burst_count = len(np.nonzero(df['Predicted BURST'])[0])
    minutes = len(df) / (sr * 60)
    return burst_count / minutes

def burst_incidence(df):
    burst_count = len(np.nonzero(df['Predicted BURST'])[0])
    ecg_count = len(ecg_peaks(df))
    return burst_count / ecg_count * 100

def burst_area(df, sr=250):
    bursts = np.nonzero(df['Predicted BURST'])[0]
    # Question: Should we use integrated or raw?
    msna = df['Integrated MSNA'].to_numpy()

    running_sum = 0

    for burst_idx in bursts:
        height = msna[burst_idx]

        # Question: Are the units on width correct?
        start = burst_idx - 1
        while msna[start] < msna[start + 1]:
            start -= 1

        end = burst_idx + 1
        while msna[end] < msna[end - 1]:
            end += 1

        width = (end - start) / sr
        running_sum += (width * height * 0.5)

    minutes = len(df) / (sr * 60)
    return running_sum / minutes