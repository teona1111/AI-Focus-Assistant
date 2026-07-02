import numpy as np

from preprocessing import (
    odstrani_dc,
    normaliziraj,
    bandpass_filter,
    preprocess_signal
)


def test_odstrani_dc_mean_is_zero():
    signal = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]
    ], dtype=float)

    result = odstrani_dc(signal)

    assert np.allclose(
        np.mean(result, axis=0),
        [0, 0, 0],
        atol=1e-8
    )


def test_normaliziraj_median_is_zero():
    signal = np.array([
        [1, 10, 100],
        [2, 20, 200],
        [3, 30, 300],
        [4, 40, 400],
        [5, 50, 500]
    ], dtype=float)

    result = normaliziraj(signal)

    assert np.allclose(
        np.median(result, axis=0),
        [0, 0, 0],
        atol=1e-8
    )


def test_normaliziraj_handles_constant_signal():
    signal = np.ones((10, 3))

    result = normaliziraj(signal)

    assert result.shape == signal.shape
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_bandpass_filter_keeps_same_shape():
    signal = np.random.randn(100, 3)

    result = bandpass_filter(
        signal,
        lowcut=0.5,
        highcut=10,
        fs=50,
        order=4
    )

    assert result.shape == signal.shape


def test_preprocess_signal_keeps_same_shape():
    signal = np.random.randn(100, 3)

    result = preprocess_signal(
        signal,
        fs=50
    )

    assert result.shape == signal.shape


def test_preprocess_signal_no_nan_or_inf():
    signal = np.random.randn(100, 3)

    result = preprocess_signal(
        signal,
        fs=50
    )

    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))