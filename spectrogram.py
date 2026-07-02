import os
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
from parser import decode_file
from analysis import sestavi_podatke
from preprocessing import preprocess_signal

WINDOW = 64
OVERLAP = 32
FFT_N = 128
ALLOWED_CLASSES = ["focus", "sleepy", "distracted"]

def normalize_to_uint8(mag):

    mag = np.log1p(mag)
    mag = mag - mag.min()

    if mag.max() > 0:
        mag = mag / mag.max()

    return (mag * 255).astype(np.uint8)

def signal_to_spectrogram(signal, Fvz, window=WINDOW, overlap=OVERLAP, fft_n=FFT_N):
    noverlap = overlap
    channels = []

    for ch in range(signal.shape[1]):
        f, t, Zxx = stft(
            signal[:, ch],
            fs=Fvz,
            nperseg=window,
            noverlap=noverlap,
            nfft=fft_n
        )
        mag = np.abs(Zxx)
        channels.append((f, t, mag))

    return channels

def spectrogram_to_rgb_image(channels_data, max_val=None):
    planes = []

    for f, t, mag in channels_data:
        plane = normalize_to_uint8(mag)
        planes.append(plane)

    rgb = np.stack(planes, axis=-1)
    return rgb

def plot_combined_rgb(channels_data, Fvz, sensor_name, filename):
    rgb = spectrogram_to_rgb_image(channels_data)
    f = channels_data[0][0]
    t = channels_data[0][1]

    plt.figure(figsize=(10, 5))
    plt.imshow(
        rgb,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], f[0], f[-1]],
    )
    plt.ylim(0, Fvz / 2)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(
        f"{sensor_name} Spectrogram\n"
        f"Fvz = {Fvz:.2f} Hz, "
        f"Window = {WINDOW}, "
        f"Overlap = {OVERLAP}, "
        f"FFT = {FFT_N}"
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

def convert_file(bin_filename, output_dir):
    paketi, _ = decode_file(bin_filename)
    base = os.path.splitext(os.path.basename(bin_filename))[0]

    sensors = [
        (1, "Gyroscope"),
        (2, "Accelerometer"),
    ]

    for sensor_id, sensor_name in sensors:
        packets = [
            p for p in paketi
            if p.id == sensor_id
        ]

        Fvz, signal = sestavi_podatke(
            packets
        )

        name = sensor_name
        unit = "value"

        if signal.size == 0:
            continue

        signal = preprocess_signal(
            signal,
            Fvz
        )
        target_seconds = 10
        target_samples = int(target_seconds * Fvz)
        signal = signal[:target_samples]
        channels_data = signal_to_spectrogram(signal, Fvz)
        filename = os.path.join(
            output_dir,
            f"{base}_{sensor_name}_spectrogram.png"
        )
        plot_combined_rgb(
            channels_data,
            Fvz,
            f"{sensor_id} - {sensor_name}",
            filename
        )

def create_folders():
    os.makedirs("spectrograms", exist_ok=True)

    for folder in ALLOWED_CLASSES:
        os.makedirs(
            os.path.join("spectrograms", folder),
            exist_ok=True
        )

def process_class(class_name):
    if class_name not in ALLOWED_CLASSES:
        print("Allowed folders:")
        print(ALLOWED_CLASSES)
        return

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

    files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(".bin")
    ]

    for file in files:
        full_path = os.path.join(input_folder, file)
        print(f"Processing: {file}")
        convert_file(full_path, output_folder)

if __name__ == "__main__":
    create_folders()
    selected_class = input("Enter class (focus/sleepy/distracted): ").strip().lower()
    process_class(selected_class)