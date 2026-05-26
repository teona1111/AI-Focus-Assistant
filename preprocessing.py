import numpy as np

from scipy.signal import (
    butter,
    filtfilt
)

from parser import decode_file
from analysis import sestavi_podatke

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

def bandpass_filter(
    signal,
    lowcut,
    highcut,
    fs,
    order=4
):

    nyq = 0.5 * fs

    if highcut >= nyq:

        highcut = nyq - 0.1

    b, a = butter_bandpass(
        lowcut,
        highcut,
        fs,
        order
    )

    filtered = np.zeros_like(
        signal
    )

    for i in range(signal.shape[1]):

        channel = signal[:, i]

        if len(channel) < 30:

            filtered[:, i] = channel

            continue

        if np.std(channel) < 1e-8:

            filtered[:, i] = channel

            continue

        try:

            filtered[:, i] = filtfilt(
                b,
                a,
                channel
            )

        except Exception:

            filtered[:, i] = channel

    return filtered


def preprocess_signal(
    signal,
    fs
):

    signal = odstrani_dc(
        signal
    )

    if fs >= 50:

        signal = bandpass_filter(
            signal,
            lowcut=0.5,
            highcut=35,
            fs=fs,
            order=4
        )

    else:

        signal = bandpass_filter(
            signal,
            lowcut=0.5,
            highcut=10,
            fs=fs,
            order=4
        )

    signal = normaliziraj(signal)

    return signal