import os
import numpy as np

from spectrogram import (
    normalize_to_uint8,
    signal_to_spectrogram,
    spectrogram_to_rgb_image,
    create_folders
)


def test_normalize_to_uint8_returns_uint8():
    mag = np.array([
        [0.1, 0.5],
        [1.0, 2.0]
    ])

    result = normalize_to_uint8(mag)

    assert result.dtype == np.uint8


def test_normalize_to_uint8_values_between_0_and_255():
    mag = np.array([
        [0.1, 0.5],
        [1.0, 2.0]
    ])

    result = normalize_to_uint8(mag)

    assert result.min() >= 0
    assert result.max() <= 255


def test_signal_to_spectrogram_returns_three_channels():
    signal = np.random.randn(200, 3)

    channels_data = signal_to_spectrogram(
        signal,
        Fvz=50
    )

    assert len(channels_data) == 3


def test_signal_to_spectrogram_channel_contains_f_t_mag():
    signal = np.random.randn(200, 3)

    channels_data = signal_to_spectrogram(
        signal,
        Fvz=50
    )

    f, t, mag = channels_data[0]

    assert f is not None
    assert t is not None
    assert mag is not None


def test_spectrogram_to_rgb_image_has_three_channels():
    signal = np.random.randn(200, 3)

    channels_data = signal_to_spectrogram(
        signal,
        Fvz=50
    )

    rgb = spectrogram_to_rgb_image(
        channels_data
    )

    assert rgb.shape[-1] == 3


def test_spectrogram_to_rgb_image_is_uint8():
    signal = np.random.randn(200, 3)

    channels_data = signal_to_spectrogram(
        signal,
        Fvz=50
    )

    rgb = spectrogram_to_rgb_image(
        channels_data
    )

    assert rgb.dtype == np.uint8


def test_create_folders_creates_spectrogram_folders(tmp_path):
    old_dir = os.getcwd()
    os.chdir(tmp_path)

    try:
        create_folders()

        assert os.path.isdir("spectrograms")
        assert os.path.isdir(os.path.join("spectrograms", "focus"))
        assert os.path.isdir(os.path.join("spectrograms", "sleepy"))
        assert os.path.isdir(os.path.join("spectrograms", "distracted"))

    finally:
        os.chdir(old_dir)