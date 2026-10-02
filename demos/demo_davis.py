"""
Minimal example showing how to use the packaged EdgeVOS API.
"""
import torch
import cv2
import numpy as np
from edgevos import EdgeVOSPredictor
from data.davis_dataset import DAVISDataset

def main():
    # 1. Initialize the unified model
    # This automatically loads DINOv3, RepViT, SpatialMaskHead, STGP, and MobileCLIP2
    model = EdgeVOSPredictor(checkpoint_path="checkpoints/edgevos_spatial_v1.pt")
    
    # 2. Get some video frames (using DAVIS dataset loader for convenience)
    print("Loading video frames...")
    dataset = DAVISDataset(root="data/DAVIS", split="val", resolution="480p")
    seq = next(s for s in dataset if s["name"] == "car-shadow")
    frames = seq["frames"].cuda()  # [T, 3, H, W]
    
    # 3. Track the object
    prompt = "car"
    print(f"\nStarting tracking for prompt: '{prompt}'")
    
    # Frame 0: Initialize from text
    mask = model.init_from_text(frames[0], text_prompt=prompt)
    print(f"Frame 0: Found initial mask of size {mask.sum().item()} pixels")
    
    # Frame 1+: Process subsequent frames
    for t in range(1, min(15, len(frames))): # Process first 15 frames for demo
        mask = model.process_frame(frames[t])
        
        if mask is not None:
            print(f"Frame {t}: Tracked object (mask size: {mask.sum().item()} pixels)")
        else:
            print(f"Frame {t}: Object lost or occluded")
            
    print("\nTracking complete! This is how easy the unified API is to use.")

if __name__ == "__main__":
    main()
