"""
Golden values from the Python reference for the Kotlin unit tests (android/core/src/test/resources/fixtures).

    python android/tools/make_fixtures.py
"""

import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT)
from mobile.runtime import grid_fraction, resize  # noqa: E402
from mobile.tokenizer import CLIPTokenizer  # noqa: E402

OUT = os.path.join(ROOT, "android", "core", "src", "test", "resources", "fixtures")

PROMPTS = ["cat", "a photo of a cat.", "a photo of a traffic light.", "Hot-Dog!!", "café 123", "it's a DOG's toy",
           "  multiple   spaces\there ", "&amp; fish &#38; chips", "über straße", "日本の猫", "person_on_bike",
           "a photo of a " + "very " * 40 + "long prompt."]


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = CLIPTokenizer()
    tokens = {p: tok(p)[0].tolist() for p in PROMPTS}

    rng = np.random.default_rng(0)
    from numpy.random import PCG64, SeedSequence
    raw = PCG64(SeedSequence(0)).random_raw(4)
    perms = [rng.permutation(n).tolist() for n in (600, 10, 1, 300)]
    rng_fx = {"seed_state": SeedSequence(0).generate_state(8, np.uint32).tolist(),
              "raw": [int(v) for v in raw], "perms": perms}

    r = np.random.default_rng(7)
    img = r.random((37, 53, 3), dtype=np.float32)
    img.tofile(os.path.join(OUT, "img_37x53x3.f32"))
    lin = {}
    for h, w in [(20, 30), (80, 100), (224, 224), (16, 28), (74, 106), (18, 26)]:
        resize(img, h, w).astype(np.float32).tofile(os.path.join(OUT, f"lin_{h}x{w}.f32"))
        lin[f"{h}x{w}"] = [h, w]
    big = r.random((250, 422), dtype=np.float32)
    big.tofile(os.path.join(OUT, "gray_250x422.f32"))
    resize(big, 36, 64, area=True).astype(np.float32).tofile(os.path.join(OUT, "area_36x64.f32"))
    resize(big[:144, :256], 36, 64, area=True).astype(np.float32).tofile(os.path.join(OUT, "area_crop_36x64.f32"))

    binary = (r.random((30, 40)) > 0.62).astype(np.uint8)
    n, labels = cv2.connectedComponents(binary, connectivity=8)
    sizes = sorted(int((labels == k).sum()) for k in range(1, n))
    mask = r.random((240, 426)) > 0.5
    mask[50:150, 100:250] = True
    frac = grid_fraction(mask, 16, 28)
    np.packbits(mask.ravel()).tofile(os.path.join(OUT, "mask_240x426.bits"))
    frac.astype(np.float32).tofile(os.path.join(OUT, "frac_16x28.f32"))
    binary.tofile(os.path.join(OUT, "binary_30x40.u8"))

    with open(os.path.join(OUT, "fixtures.json"), "w") as f:
        json.dump({"tokens": tokens, "rng": rng_fx, "linear": lin, "cc": {"n": int(n), "sizes": sizes}}, f)
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
