import numpy as np

from analysis import sestavi_podatke


class MockPacket:
    def __init__(self, ts, data):
        self.ts = ts
        self.data = data


def test_sestavi_podatke_empty_list():
    fvz, signal = sestavi_podatke([])

    assert fvz == 0
    assert isinstance(signal, np.ndarray)
    assert signal.size == 0


def test_sestavi_podatke_ignores_empty_packets():
    packets = [
        MockPacket(0, []),
        MockPacket(10, None),
        MockPacket(20, np.array([[1, 2, 3]]))
    ]

    fvz, signal = sestavi_podatke(packets)

    assert signal.shape == (1, 3)


def test_sestavi_podatke_combines_packet_data():
    packets = [
        MockPacket(0, np.array([[1, 2, 3], [4, 5, 6]])),
        MockPacket(10, np.array([[7, 8, 9], [10, 11, 12]]))
    ]

    fvz, signal = sestavi_podatke(packets)

    expected = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])

    assert np.array_equal(signal, expected)


def test_sestavi_podatke_sampling_frequency_positive():
    packets = [
        MockPacket(0, np.array([[1, 2, 3], [4, 5, 6]])),
        MockPacket(10, np.array([[7, 8, 9], [10, 11, 12]])),
        MockPacket(20, np.array([[13, 14, 15], [16, 17, 18]]))
    ]

    fvz, signal = sestavi_podatke(packets)

    assert fvz > 0


def test_sestavi_podatke_sampling_frequency_value():
    packets = [
        MockPacket(0, np.array([[1, 2, 3], [4, 5, 6]])),
        MockPacket(10, np.array([[7, 8, 9], [10, 11, 12]])),
        MockPacket(20, np.array([[13, 14, 15], [16, 17, 18]]))
    ]

    fvz, signal = sestavi_podatke(packets)

    assert np.isclose(fvz, 0.2)