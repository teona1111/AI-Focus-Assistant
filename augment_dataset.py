import os
import numpy as np

from parser import decode_file
from analysis import sestavi_podatke
from preprocessing import preprocess_signal

from spectrogram import (
    signal_to_spectrogram,
    plot_combined_rgb
)

ALLOWED_CLASSES = [
    "focus",
    "sleepy",
    "distracted"
]

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

def add_noise(signal):

    noise = np.random.normal(
        0,
        0.01 * np.std(signal),
        signal.shape
    )

    return signal + noise


def scale_signal(signal):

    scale = np.random.uniform(
        0.95,
        1.05
    )

    return signal * scale


def channel_scale(signal):

    scales = np.random.uniform(
        0.95,
        1.05,
        size=(1, signal.shape[1])
    )

    return signal * scales


def shift_signal(signal):

    max_shift = int(
        len(signal) * 0.10
    )

    if max_shift < 1:
        return signal

    shift = np.random.randint(
        -max_shift,
        max_shift
    )

    return np.roll(
        signal,
        shift,
        axis=0
    )


def crop_and_resize(signal):

    N = len(signal)

    crop_ratio = np.random.uniform(
        0.80,
        0.95
    )

    crop_len = int(
        N * crop_ratio
    )

    start = np.random.randint(
        0,
        N - crop_len
    )

    cropped = signal[
        start:start + crop_len
    ]

    old_x = np.linspace(
        0,
        1,
        len(cropped)
    )

    new_x = np.linspace(
        0,
        1,
        N
    )

    resized = np.zeros_like(
        signal
    )

    for ch in range(signal.shape[1]):

        resized[:, ch] = np.interp(
            new_x,
            old_x,
            cropped[:, ch]
        )

    return resized

def save_augmented_spectrogram(
    signal,
    Fvz,
    filename,
    sensor_name
):

    channels_data = signal_to_spectrogram(
        signal,
        Fvz
    )

    plot_combined_rgb(
        channels_data,
        Fvz,
        sensor_name,
        filename
    )

def process_file(
    filepath,
    output_folder
):

    paketi, _ = decode_file(
        filepath
    )

    sensors = [
        (1, "gyro"),
        (2, "acc")
    ]

    base = os.path.splitext(os.path.basename(filepath))[0]

    for sensor_id, sensor_name in sensors:

        packets = [
            p for p in paketi
            if p.id == sensor_id
        ]

        Fvz, signal = sestavi_podatke(packets)

        if len(signal) == 0:
            continue

        signal = preprocess_signal(signal, Fvz)
        target_samples = int(Fvz * 10)
        signal = signal[:target_samples]

        versions = [
            ("orig", signal),
            ("noise", add_noise(signal)),
            ("scale", scale_signal(signal)),
            ("chscale", channel_scale(signal)),
            ("shift", shift_signal(signal)),
            ("crop", crop_and_resize(signal))
        ]

        for aug_name, aug_signal in versions:

            filename = os.path.join(
                output_folder,
                f"{base}_{aug_name}_{sensor_name}_spektrogram.png"
            )

            save_augmented_spectrogram(
                aug_signal,
                Fvz,
                filename,
                sensor_name
            )

def main():

    for class_name in ALLOWED_CLASSES:

        print()
        print(f"Processing {class_name}")

        input_folder = os.path.join(
            BASE_DIR,
            "data",
            class_name
        )

        output_folder = os.path.join(
            BASE_DIR,
            "spectrograms",
            class_name
        )

        os.makedirs(
            output_folder,
            exist_ok=True
        )

        files = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith(".bin")
        ]

        for file in files:

            full_path = os.path.join(input_folder, file)
            print("Processing:", file)
            process_file(full_path, output_folder)

    print()
    print("Augmentation finished.")

if __name__ == "__main__":
    main()