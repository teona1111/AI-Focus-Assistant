import numpy as np

def normalize_to_uint8(mag):

    mag = np.log1p(mag)

    mag = mag - mag.min()

    if mag.max() > 0:
        mag = mag / mag.max()

    return (mag * 255).astype(np.uint8)