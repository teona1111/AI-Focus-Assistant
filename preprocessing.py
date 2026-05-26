import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import (
    butter,
    filtfilt
)

from parser import decode_file
from analysis import sestavi_podatke

import os 
file_path = os.path.join("data" , "sleepy", "1.bin")


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

if __name__ == "__main__":

    paketi, raw_packets = decode_file(
        file_path
    )

    sensors = [
        1,
        2
    ]

    osi = [
        "X",
        "Y",
        "Z"
    ]

    for sensor_id in sensors:

        packets = [
            p for p in paketi
            if p.id == sensor_id
        ]

        fvz, signal_data = sestavi_podatke(
            packets
        )

        if sensor_id == 1:

            name = "Gyroscope"

            unit = "deg/s"

        else:

            name = "Accelerometer"

            unit = "m/sÂ²"

        processed_signal = preprocess_signal(
            signal_data,
            fvz
        )
        

        samples = len(
            signal_data
        )

        time = np.arange(
            samples
        ) / fvz

        fig, axs = plt.subplots(
            3,
            1,
            figsize=(14, 10)
        )

        for i in range(3):

            axs[i].plot(time,signal_data[:samples, i],label="Original")

            axs[i].plot(
                time,
                processed_signal[:samples, i],
                label="Processed"
            )

            axs[i].set_title(
                f"{osi[i]} axis"
            )

            axs[i].set_xlabel(
                "Time (s)"
            )

            axs[i].set_ylabel(
                unit
            )

            axs[i].grid()

            axs[i].legend()

        plt.suptitle(
            f"Preprocessing - {name} "
            f"(Fvz={fvz:.2f} Hz)"
        )

        plt.tight_layout()

        plt.savefig(
            f"{name}_preprocessing.png",
            dpi=300
        )

    plt.show()