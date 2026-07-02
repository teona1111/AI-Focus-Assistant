import struct
import numpy as np

class Paket:
    def __init__(self, id, ts, data):
        self.id = id
        self.ts = ts
        self.data = data

def crc16_update(crc, data):
    crc ^= data
    for _ in range(8):
        if crc & 1:
            crc = (crc >> 1) ^ 0xA001
        else:
            crc >>= 1
    return crc


def crc16_compute(data):
    crc = 0xFFFF
    for b in data:
        crc = crc16_update(crc, b)
    return crc

def unstuff_bytes(data):
    result = bytearray()
    i = 0

    while i < len(data):
        if data[i] == 0xFE:
            if i + 1 >= len(data):
                break
            result.append(0xFE ^ data[i + 1])
            i += 2
        else:
            result.append(data[i])
            i += 1

    return bytes(result)

def parse_packet(raw_packet):
    payload = unstuff_bytes(raw_packet[1:])

    if len(payload) < 8:
        return None

    timestamp = struct.unpack('<I', payload[0:4])[0]

    crc_recv = struct.unpack('<H', payload[-2:])[0]
    crc_calc = crc16_compute(payload[:-2])

    if crc_recv != crc_calc:
        return None

    pos = 6
    end = len(payload) - 2

    paketi = []

    while pos < end:
        cid = payload[pos]
        size = struct.unpack('<H', payload[pos+1:pos+3])[0] + 1

        start = pos + 4
        stop = start + size

        chunk = payload[start:stop]

        samples = []
        for i in range(0, len(chunk), 6):
            if i + 6 > len(chunk):
                break
            x, y, z = struct.unpack('<hhh', chunk[i:i+6])
            samples.append((x, y, z))

        data = np.array(samples, dtype=np.int16)

        paketi.append(Paket(cid, timestamp / 1000.0, data))
        pos = stop

    return paketi

def extract_packets(data):
    packets = []
    i = 0

    while i < len(data) - 3:
        if data[i] == 0xFF and data[i+1] == 0xFF:
            j = i + 2
            while j < len(data) - 1:
                if data[j] == 0xFF and data[j+1] == 0xFF:
                    break
                j += 1

            packets.append(data[i+2:j])
            i = j
        else:
            i += 1

    return packets


def decode_file(filename):
    with open(filename, "rb") as f:
        data = f.read()

    raw = extract_packets(data)

    vsi = []
    for p in raw:
        parsed = parse_packet(p)
        if parsed:
            vsi.extend(parsed)

    return vsi, raw