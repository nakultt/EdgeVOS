"""
Dump reference masks + events from mobile/runtime.py (the Python MobileTracker) for parity checks.

    python android/tools/parity_reference.py --video b.mp4 --prompt cat --out parity/py_cat

Frames are decoded with the ffmpeg CLI (rgb24) so the Kotlin runner (`:core:parity`), which
uses the same command, sees byte-identical input. The model setup mirrors the app: fp32 vision
models and the INT8 text encoder (pass --fp32-text for the all-fp32 setup).

Use the ONNX Runtime version the app ships (pip install -r android/tools/requirements.txt):
the INT8 text encoder's output differs between ORT releases.

Writes <out>/masks.bin (see android/tools/masks.py) and <out>/events.tsv (frame, event, ms).
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from masks import MaskWriter  # noqa: E402
from mobile import runtime  # noqa: E402
from mobile.runtime import MobileTracker  # noqa: E402

APP_ORT = "1.29.0"   # onnxruntime-android-qnn in android/app/build.gradle.kts


def probe_size(video):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height", "-of", "csv=p=0", video], text=True)
    w, h = out.strip().split(",")[:2]
    return int(h), int(w)


def frames(video, H, W, limit=None):
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE)
    n, size = 0, H * W * 3
    try:
        while limit is None or n < limit:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            n += 1
    finally:
        proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="mobile/models")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fp32-text", action="store_true")
    args = ap.parse_args()

    if ort.__version__ != APP_ORT:
        print(f"WARNING: onnxruntime {ort.__version__}, the app uses {APP_ORT}; INT8 text results will differ slightly")
    tracker = MobileTracker(args.models)
    if not args.fp32_text:   # the app runs the INT8 text encoder on the CPU
        so = ort.SessionOptions()
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        tracker.models.clip_text = ort.InferenceSession(os.path.join(args.models, "clip_text.int8.onnx"), so,
                                                        providers=["CPUExecutionProvider"])

    # Count how often PatchMemory subsamples a bank (the only random step in the tracker).
    caps = [0]
    orig_cap = runtime.PatchMemory._cap

    def counting_cap(self, x, n):
        if len(x) > n:
            caps[0] += 1
        return orig_cap(self, x, n)

    runtime.PatchMemory._cap = counting_cap

    H, W = probe_size(args.video)
    os.makedirs(args.out, exist_ok=True)
    writer = MaskWriter(os.path.join(args.out, "masks.bin"), H, W)
    with open(os.path.join(args.out, "events.tsv"), "w") as ev:
        start = time.time()
        for i, frame in enumerate(frames(args.video, H, W, args.limit)):
            t = time.perf_counter()
            mask = tracker.init(frame, args.prompt) if i == 0 else tracker.step(frame)
            ms = (time.perf_counter() - t) * 1000
            writer.write(mask)
            ev.write(f"{i}\t{tracker.last_event or '-'}\t{ms:.1f}\n")
            if tracker.last_event:
                print(i, tracker.last_event, flush=True)
    writer.close()
    print(f"{i + 1} frames in {time.time() - start:.0f} s; bank subsampling events: {caps[0]}")


if __name__ == "__main__":
    main()
