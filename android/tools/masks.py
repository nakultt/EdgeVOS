"""
masks.bin: the per-frame mask format shared by the Python reference, the JVM parity runner
and the app's on-device parity mode.

    header: b"EVM1", int32 LE height, int32 LE width
    per frame: 1 byte (0 = no mask / absent, 1 = mask follows) + H*W bits, row-major,
               packed MSB-first (numpy.packbits), padded to a whole byte
"""

import struct

import numpy as np


class MaskWriter:
    def __init__(self, path, H, W):
        self.f = open(path, "wb")
        self.H, self.W = H, W
        self.f.write(b"EVM1" + struct.pack("<ii", H, W))

    def write(self, mask):
        if mask is None:
            self.f.write(b"\0")
        else:
            assert mask.shape == (self.H, self.W)
            self.f.write(b"\1" + np.packbits(mask.astype(bool).ravel()).tobytes())

    def close(self):
        self.f.close()


def read_masks(path):
    """→ (H, W, list of [H, W] bool arrays or None)."""
    data = open(path, "rb").read()
    assert data[:4] == b"EVM1", path
    H, W = struct.unpack("<ii", data[4:12])
    nbytes = (H * W + 7) // 8
    out, p = [], 12
    while p < len(data):
        flag = data[p]
        p += 1
        if flag:
            bits = np.unpackbits(np.frombuffer(data, np.uint8, nbytes, p))[:H * W]
            out.append(bits.reshape(H, W).astype(bool))
            p += nbytes
        else:
            out.append(None)
    return H, W, out
