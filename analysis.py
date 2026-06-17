import matplotlib.pyplot as plt
import numpy as np

from parser import decode_file


def sestavi_podatke(seznam_paketov):

    all_data = []
    timestamps = []

    for p in seznam_paketov:

        if p.data is None or len(p.data) == 0:
            continue

        all_data.append(p.data)
        timestamps.append(p.ts)

    if len(all_data) == 0:
        return 0, np.array([])

    signal = np.vstack(all_data)

    timestamps = np.array(
        timestamps,
        dtype=float
    )

    if len(timestamps) < 2:
        return 0, signal

    dt = np.diff(timestamps)

    dt = dt[dt > 0]

    if len(dt) == 0:
        return 0, signal

    Tavg = np.mean(dt)

    samples_per_packet = np.mean([
        len(p.data)
        for p in seznam_paketov
        if p.data is not None and len(p.data) > 0
    ])

    Fvz = samples_per_packet / Tavg

    return Fvz, signal


def prikazi_signal(
    signal,
    Fvz,
    naslov="",
    ylabel="",
    startInd=None,
    endInd=None
):

    if len(signal) == 0:
        print("Ni podatkov.")
        return

    if Fvz <= 0:
        print("Napacna frekvenca vzorcenja.")
        return

    if startInd is not None and endInd is not None:
        signal = signal[startInd:endInd]

    t = np.arange(len(signal)) / Fvz

    plt.figure(figsize=(10, 4))

    plt.plot(
        t,
        signal[:, 0],
        label="x"
    )

    plt.plot(
        t,
        signal[:, 1],
        label="y"
    )

    plt.plot(
        t,
        signal[:, 2],
        label="z"
    )

    plt.title(
        f"{naslov} (Fvz = {Fvz:.2f} Hz)"
    )

    plt.xlabel("time (s)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def main():

    file_path = input(
        "Enter the .bin file to analyze: "
    )

    try:

        paketi, _ = decode_file(
            file_path
        )

    except Exception as e:

        print(
            f"Error while opening file: {e}"
        )

        return

    gyro_packets = [
        p for p in paketi
        if p.id == 1
    ]

    accel_packets = [
        p for p in paketi
        if p.id == 2
    ]

    Fg, gyro = sestavi_podatke(
        gyro_packets
    )

    Fa, accel = sestavi_podatke(
        accel_packets
    )

    gyro = gyro * 8.75e-3
    accel = accel * 6.125e-5

    print("Gyro Fvz:", Fg)
    print("Accel Fvz:", Fa)

    prikazi_signal(
        gyro,
        Fg,
        "Gyroscope",
        "deg/s"
    )

    prikazi_signal(
        accel,
        Fa,
        "Accelerometer",
        "g"
    )


if __name__ == "__main__":
    main()