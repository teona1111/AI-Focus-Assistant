import numpy as np
from scipy.signal import stft

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