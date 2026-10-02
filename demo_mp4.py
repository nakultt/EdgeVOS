"""
EdgeVOS MP4 Video Processor
Loads a custom MP4 file, tracks an object via text prompt, and saves the output video.

Usage:
    python demo_mp4.py --video_path path/to/video.mp4 --prompt "object to track" --output output.mp4
"""

import argparse
import cv2
from tqdm import tqdm
from edgevos import EdgeVOSPredictor

def main():
    parser = argparse.ArgumentParser(description="Run EdgeVOS on a custom MP4 video.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input .mp4 file")
    parser.add_argument("--prompt", type=str, required=True, help="What object to track (e.g. 'car')")
    parser.add_argument("--output", type=str, default="tracked_output.mp4", help="Output video path")
    parser.add_argument("--ckpt", type=str, default="checkpoints/edgevos_spatial_v1.pt", help="Checkpoint path")
    parser.add_argument("--use_sam3", action="store_true", help="Dev only: ground with the SAM 3 teacher")
    args = parser.parse_args()

    # 1. Load the model
    model = EdgeVOSPredictor(checkpoint_path=args.ckpt, use_sam3=args.use_sam3)

    # 2. Open the video file
    video = cv2.VideoCapture(args.video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {args.video_path}")
        return

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS) or 30
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"\nProcessing {args.video_path} ({total_frames} frames at {fps} fps)")
    print(f"Tracking object: '{args.prompt}'")

    # 3. Read frame 0 and initialise (frame_to_tensor: OpenCV BGR -> RGB tensor on device)
    ret, frame_bgr = video.read()
    if not ret:
        print("Error reading first frame.")
        return
    mask = model.init_from_text(model.frame_to_tensor(frame_bgr), text_prompt=args.prompt)
    if mask is None:
        print(f"'{args.prompt}' not visible in the first frame; will keep searching.")

    # 4. Track. Green = (re)grounded frame, red = tracked frame, no overlay = absent.
    with tqdm(total=total_frames, desc="Tracking") as pbar:
        while ret:
            if mask is not None:
                overlay = frame_bgr.copy()
                grounded = (model.last_event or "").startswith(("init", "reground"))
                overlay[mask.cpu().numpy()] = [0, 255, 0] if grounded else [0, 0, 255]
                frame_bgr = cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0)
            out_video.write(frame_bgr)
            pbar.update(1)

            ret, frame_bgr = video.read()
            if ret:
                mask = model.process_frame(model.frame_to_tensor(frame_bgr))

    video.release()
    out_video.release()
    print(f"\nDone! Video saved to {args.output}")

if __name__ == "__main__":
    main()
