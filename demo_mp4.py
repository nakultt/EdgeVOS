"""
EdgeVOS MP4 Video Processor
Loads a custom MP4 file, tracks an object via text prompt, and saves the output video.

Usage:
    python demo_mp4.py --video_path path/to/video.mp4 --prompt "object to track" --output output.mp4
"""

import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm
from edgevos import EdgeVOSPredictor

def main():
    parser = argparse.ArgumentParser(description="Run EdgeVOS on a custom MP4 video.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input .mp4 file")
    parser.add_argument("--prompt", type=str, required=True, help="What object to track (e.g. 'car')")
    parser.add_argument("--output", type=str, default="tracked_output.mp4", help="Output video path")
    parser.add_argument("--ckpt", type=str, default="checkpoints/edgevos_spatial_v1.pt", help="Checkpoint path")
    args = parser.parse_args()

    # 1. Load the model
    model = EdgeVOSPredictor(checkpoint_path=args.ckpt)

    # 2. Open the video file
    video = cv2.VideoCapture(args.video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {args.video_path}")
        return

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"\nProcessing {args.video_path} ({total_frames} frames at {fps} fps)")
    print(f"Tracking object: '{args.prompt}'")

    # 3. Read Frame 0 and Initialize
    ret, frame_bgr = video.read()
    if not ret:
        print("Error reading first frame.")
        return

    # Convert BGR (OpenCV) -> RGB -> PyTorch Tensor [3, H, W]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).cuda().float() / 255.0

    mask = model.init_from_text(frame_tensor, text_prompt=args.prompt)
    
    # Draw initial mask on frame 0 (Green)
    overlay = frame_bgr.copy()
    overlay[mask.cpu().numpy()] = [0, 255, 0]
    out_video.write(cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0))

    # 4. Loop through the rest of the video
    for _ in tqdm(range(1, total_frames), desc="Tracking"):
        ret, frame_bgr = video.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).cuda().float() / 255.0

        mask = model.process_frame(frame_tensor)

        # Draw mask
        if mask is not None:
            mask_np = mask.cpu().numpy()
            overlay = frame_bgr.copy()
            overlay[mask_np] = [0, 0, 255] # Red for tracked frames
            frame_bgr = cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0)
        
        out_video.write(frame_bgr)

    video.release()
    out_video.release()
    print(f"\nDone! Video saved to {args.output}")

if __name__ == "__main__":
    main()
