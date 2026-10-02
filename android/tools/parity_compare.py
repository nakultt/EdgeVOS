"""
Compare two tracker runs (masks.bin + events.tsv), e.g. Kotlin (JVM or on-device) vs mobile/runtime.py.

    python android/tools/parity_compare.py /path/to/py_cat /path/to/kt_cat [--min-iou 0.95]

Per-frame IoU: both absent → 1, one absent → 0. Prints mean IoU (all frames, and frames where
either run has a mask), presence agreement and every event that differs. Exits non-zero below --min-iou
or when the event sequences differ.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from masks import read_masks  # noqa: E402


def events(path):
    out = {}
    for line in open(os.path.join(path, "events.tsv")):
        f = line.rstrip("\n").split("\t")
        out[int(f[0])] = f[1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--min-iou", type=float, default=0.95)
    args = ap.parse_args()

    H, W, ref = read_masks(os.path.join(args.ref, "masks.bin"))
    H2, W2, test = read_masks(os.path.join(args.test, "masks.bin"))
    assert (H, W) == (H2, W2), ((H, W), (H2, W2))
    n = min(len(ref), len(test))
    if len(ref) != len(test):
        print(f"WARNING: frame counts differ: ref {len(ref)}, test {len(test)}; comparing {n}")
    ious, either, worst = [], [], []
    for i in range(n):
        a, b = ref[i], test[i]
        if a is None and b is None:
            iou = 1.0
        elif a is None or b is None:
            iou = 0.0
        else:
            union = (a | b).sum()
            iou = float((a & b).sum() / union) if union else 1.0
        ious.append(iou)
        if a is not None or b is not None:
            either.append(iou)
        worst.append((iou, i))
    ious = np.array(ious)
    present_agree = np.mean([(ref[i] is None) == (test[i] is None) for i in range(n)])
    ev_ref, ev_test = events(args.ref), events(args.test)
    diffs = [(i, ev_ref.get(i, "-"), ev_test.get(i, "-")) for i in range(n) if ev_ref.get(i, "-") != ev_test.get(i, "-")]

    print(f"frames: {n} ({H}x{W})")
    print(f"mean IoU (all frames): {ious.mean():.4f}")
    if either:
        print(f"mean IoU (frames with a mask in either run): {np.mean(either):.4f} over {len(either)} frames")
    print(f"presence agreement: {present_agree:.4f}; frames with a mask: ref {sum(m is not None for m in ref[:n])}, "
          f"test {sum(m is not None for m in test[:n])}")
    absent = sum(e == "absent" for e in ev_ref.values())
    print("ref events:", [(i, e) for i, e in sorted(ev_ref.items()) if e not in ("-", "absent")], f"+ {absent} 'absent'")
    print("event differences:", diffs if diffs else "none")
    print("worst frames:", [(i, round(v, 3)) for v, i in sorted(worst)[:8]])
    ok = ious.mean() >= args.min_iou and not diffs
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
