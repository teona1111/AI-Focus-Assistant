import serial
import time
import threading
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from parser import extract_packets, parse_packet
from analysis import sestavi_podatke

PORT = "COM7"
BAUD = 115200
CROP_SECONDS = 1.5

allowed_labels = ["focus", "distracted", "sleepy"]

while True:
    label = input("Enter label (focus/distracted/sleepy): ").lower()

    if label in allowed_labels:
        break

    print("Invalid label. Use: focus, distracted, or sleepy.")

folder = f"data/{label}"
os.makedirs(folder, exist_ok=True)

i = 1
while True:
    filename = f"{folder}/{i}.bin"

    if not os.path.exists(filename):
        break

    i += 1

print(f"[SAVE] Recording to {filename}")

buffer = b""
paketi = []
running = True


def read_serial():
    global buffer, paketi

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        print(f"[OK] Connected to {PORT}")

    except Exception as e:
        print("[ERROR] Serial failed:", e)
        return

    raw_chunks = []
    start_time = None

    while running:

        data = ser.read(256)

        if data:

            if start_time is None:
                start_time = time.time()


            raw_chunks.append((time.time(), data))

            buffer += data

            raw_packets = extract_packets(buffer)

            if raw_packets:
                last_packet = raw_packets[-1]
                last_index = buffer.rfind(last_packet)

                buffer = buffer[last_index + len(last_packet):]

            for rp in raw_packets:
                parsed = parse_packet(rp)
                paketi.extend(parsed)

    ser.close()

    if raw_chunks and start_time is not None:

        end_time = raw_chunks[-1][0]

        with open(filename, "wb") as f:

            for t_chunk, chunk in raw_chunks:

                if (
                    t_chunk >= start_time + CROP_SECONDS
                    and t_chunk <= end_time - CROP_SECONDS
                ):
                    f.write(chunk)


def live_plot():

    print("[OK] Starting plot window...")

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    while running:

        if len(paketi) < 5:
            plt.pause(0.1)
            continue

        for ax in axs:
            ax.clear()

        for idx, sensor_id in enumerate([1, 2, 3]):

            try:
                Fvz, sig, name, unit = sestavi_podatke(paketi, sensor_id)

                if sig.size == 0 or Fvz == 0:
                    continue

                s = sig

                t = np.arange(len(s)) / Fvz

                axs[idx].plot(t, s[:, 0], label="x")
                axs[idx].plot(t, s[:, 1], label="y")
                axs[idx].plot(t, s[:, 2], label="z")

                axs[idx].set_title(f"{name} (Fvz={Fvz:.1f} Hz)")
                axs[idx].set_ylabel(unit)

                axs[idx].legend()
                axs[idx].grid()

            except Exception as e:
                print("[PLOT ERROR]", e)

        axs[-1].set_xlabel("time (s)")

        plt.tight_layout()

        plt.pause(0.05)

    plt.close(fig)


if __name__ == "__main__":

    print("[START] Program running...")

    t1 = threading.Thread(target=read_serial)
    #t2 = threading.Thread(target=live_plot)

    t1.start()
    #t2.start()

    try:
        live_plot()

    except KeyboardInterrupt:

        print("\n[STOP] Closing...")

        running = False

        t1.join()
        #t2.join()
        plt.close('all')


        if os.path.exists(filename):

            if os.path.getsize(filename) == 0:
                os.remove(filename)
                print("[INFO] Empty recording discarded.")

            else:
                print(f"[OK] Saved {filename}")