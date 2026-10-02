"""
Run the on-device pipeline (ONNX Runtime, no PyTorch) on a video file.

    python -m mobile.demo --video b.mp4 --prompt cat --output tracked_mobile.mp4 [--int8]

Needs exported models: python scripts/export_mobile.py --int8 --ort
"""

import argparse
import time

import cv2

from mobile.runtime import MobileTracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--output", default="tracked_mobile.mp4")
    ap.add_argument("--models", default="mobile/models")
    ap.add_argument("--int8", action="store_true", help="use INT8 DINOv3 / text encoder")
    args = ap.parse_args()

    tracker = MobileTracker(args.models, int8=args.int8)
    video = cv2.VideoCapture(args.video)
    W, H = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), video.get(cv2.CAP_PROP_FPS) or 30, (W, H))

    i, start = 0, time.time()
    while True:
        ok, frame = video.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = tracker.init(rgb, args.prompt) if i == 0 else tracker.step(rgb)
        if mask is not None:
            overlay = frame.copy()
            grounded = (tracker.last_event or "").startswith(("init", "reground"))
            overlay[mask] = (0, 255, 0) if grounded else (0, 0, 255)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        out.write(frame)
        i += 1
        if i % 100 == 0:
            print(f"{i} frames, {i / (time.time() - start):.1f} fps")
    out.release()
    print(f"Saved {args.output} ({i} frames, {i / (time.time() - start):.1f} fps)")


if __name__ == "__main__":
    main()
