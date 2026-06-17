import time
import joblib
import serial
import numpy as np

from PIL import Image

from parser import extract_packets, parse_packet
from analysis import sestavi_podatke
from preprocessing import preprocess_signal

from spectrogram import (
    signal_to_spectrogram,
    spectrogram_to_rgb_image
)

PORT = "COM7"
BAUD = 115200

MODEL_PATH = "rf_combined.pkl"

CLASSES = [
    "focus",
    "sleepy",
    "distracted"
]


class SensorPredictor:

    def __init__(self):

        self.model = joblib.load(
            MODEL_PATH
        )

    def crop_or_pad(
        self,
        signal,
        fs,
        seconds=10
    ):

        target = int(
            fs * seconds
        )

        if len(signal) > target:

            return signal[:target]

        if len(signal) < target:

            missing = target - len(signal)

            padding = np.zeros(
                (
                    missing,
                    signal.shape[1]
                )
            )

            signal = np.vstack(
                [
                    signal,
                    padding
                ]
            )

        return signal

    def prepare_image(
        self,
        rgb
    ):

        image = Image.fromarray(
            rgb
        )

        image = image.convert(
            "RGBA"
        )

        image = image.resize(
            (64, 64),
            Image.Resampling.BILINEAR
        )

        image = np.array(
            image
        )

        return image.flatten()

    def record_packets(
        self,
        seconds=10
    ):

        ser = serial.Serial(
            PORT,
            BAUD,
            timeout=1
        )

        buffer = b""

        packets = []

        start = time.time()

        while time.time() - start < seconds:

            data = ser.read(
                2048
            )

            if not data:
                continue

            buffer += data

            raw_packets = extract_packets(
                buffer
            )

            if raw_packets:

                last_packet = raw_packets[-1]

                last_index = buffer.rfind(
                    last_packet
                )

                buffer = buffer[
                    last_index + len(last_packet):
                ]

            for rp in raw_packets:

                parsed = parse_packet(
                    rp
                )

                if parsed:

                    packets.extend(
                        parsed
                    )

        ser.close()

        return packets

    def predict(
        self
    ):

        packets = self.record_packets()

        gyro_packets = [
            p for p in packets
            if p.id == 1
        ]

        acc_packets = [
            p for p in packets
            if p.id == 2
        ]

        if len(gyro_packets) == 0:

            return None

        if len(acc_packets) == 0:

            return None

        Fg, gyro = sestavi_podatke(
            gyro_packets
        )

        Fa, acc = sestavi_podatke(
            acc_packets
        )

        gyro = preprocess_signal(
            gyro,
            Fg
        )

        acc = preprocess_signal(
            acc,
            Fa
        )

        gyro = self.crop_or_pad(
            gyro,
            Fg,
            10
        )

        acc = self.crop_or_pad(
            acc,
            Fa,
            10
        )

        gyro_spec = signal_to_spectrogram(
            gyro,
            Fg
        )

        acc_spec = signal_to_spectrogram(
            acc,
            Fa
        )

        gyro_rgb = spectrogram_to_rgb_image(
            gyro_spec
        )

        acc_rgb = spectrogram_to_rgb_image(
            acc_spec
        )

        gyro_x = self.prepare_image(
            gyro_rgb
        )

        acc_x = self.prepare_image(
            acc_rgb
        )

        features = np.concatenate(
            [
                gyro_x,
                acc_x
            ]
        )

        prob = self.model.predict_proba(
            [
                features
            ]
        )[0]

        return {
            "focus": float(prob[0]),
            "sleepy": float(prob[1]),
            "distracted": float(prob[2])
        }


if __name__ == "__main__":

    predictor = SensorPredictor()

    result = predictor.predict()

    print(result)