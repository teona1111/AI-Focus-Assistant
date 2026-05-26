import numpy as np

from scipy.signal import (
    butter,
    filtfilt
)

from parser import decode_file
from analysis import sestavi_podatke
file_path = r"C:\Users\filip\Desktop\nova_data\data\sleepy\1.BIN"

def odstrani_dc(signal):

    return signal - np.mean(
        signal,
        axis=0
    )

def normaliziraj(signal):

    median = np.median(
        signal,
        axis=0
    )

    q75 = np.percentile(
        signal,
        75,
        axis=0
    )

    q25 = np.percentile(
        signal,
        25,
        axis=0
    )

    iqr = q75 - q25
    iqr[iqr == 0] = 1
    return (signal - median) / iqr


def butter_bandpass(
    lowcut,
    highcut,
    fs,
    order=4
):

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    return butter(
        order,
        [low, high],
        btype='band'
    )