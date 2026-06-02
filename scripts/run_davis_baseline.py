"""
EdgeVOS — Month 2: Optical Flow Baseline on DAVIS-2017
Establishes the Farneback optical flow and copy-mask baselines.

Run:
  python scripts/run_davis_baseline.py
"""

import os
import sys
import json
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.davis_dataset import DAVISDataset
from eval.davis_eval import jaccard, f_measure


def warp_mask_optical_flow(prev_mask: np.ndarray, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """Warp mask from t-1 to t using Farneback optical flow calculated from curr to prev."""
    # Compute backward flow so we can do backward warping without holes
    flow = cv2.calcOpticalFlowFarneback(
        prev = curr_gray,
        next = prev_gray,
        flow = None,
        pyr_scale = 0.5,
        levels = 3,
        winsize = 15,
        iterations = 3,
        poly_n = 5,
        poly_sigma = 1.2,
        flags = 0
    )
    
    h, w = prev_mask.shape
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    
    # Meshgrid for remap coordinates
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow_x).astype(np.float32)
    map_y = (grid_y + flow_y).astype(np.float32)
    
    # Warp mask
    warped_mask = cv2.remap(
        prev_mask.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return (warped_mask > 0.5).astype(np.uint8)


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    # ── Setup ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading DAVIS val dataset from {cfg.data.davis_root}...")
    
    dataset = DAVISDataset(
        root=cfg.data.davis_root,
        split="val",
        resolution=cfg.eval.davis_resolution,
        img_size=cfg.data.img_size,
    )

    results_flow = {}
    results_copy = {}
    
    all_J_flow, all_F_flow = [], []
    all_J_copy, all_F_copy = [] ,[]

    for idx in tqdm(range(len(dataset)), desc="Evaluating Baselines"):
        seq = dataset[idx]
        name = seq["name"]
        frames = seq["frames"]       # [T, 3, H, W]
        gt_masks = seq["masks"].numpy().astype(np.uint8)  # [T, H, W]
        T = seq["n_frames"]

        seq_J_flow, seq_F_flow = [1.0], [1.0]
        seq_J_copy, seq_F_copy = [1.0], [1.0]

        # Convert frame 0 to grayscale for optical flow init
        frame_0 = (frames[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        prev_gray = cv2.cvtColor(frame_0, cv2.COLOR_RGB2GRAY)
        
        prev_mask_flow = gt_masks[0]
        prev_mask_copy = gt_masks[0]

        for t in range(1, T):
            curr_frame = (frames[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            gt_mask = gt_masks[t]

            # 1. Optical Flow Propagation
            pred_mask_flow = warp_mask_optical_flow(prev_mask_flow, prev_gray, curr_gray)
            seq_J_flow.append(jaccard(pred_mask_flow, gt_mask))
            seq_F_flow.append(f_measure(pred_mask_flow, gt_mask))

            # 2. Copy Previous Mask Baseline
            pred_mask_copy = prev_mask_copy
            seq_J_copy.append(jaccard(pred_mask_copy, gt_mask))
            seq_F_copy.append(f_measure(pred_mask_copy, gt_mask))

            # Update history
            prev_gray = curr_gray
            prev_mask_flow = pred_mask_flow
            prev_mask_copy = pred_mask_copy

        mean_J_flow = np.mean(seq_J_flow)
        mean_F_flow = np.mean(seq_F_flow)
        mean_JF_flow = (mean_J_flow + mean_F_flow) / 2
        results_flow[name] = {"J": mean_J_flow, "F": mean_F_flow, "JF": mean_JF_flow}
        all_J_flow.append(mean_J_flow)
        all_F_flow.append(mean_F_flow)

        mean_J_copy = np.mean(seq_J_copy)
        mean_F_copy = np.mean(seq_F_copy)
        mean_JF_copy = (mean_J_copy + mean_F_copy) / 2
        results_copy[name] = {"J": mean_J_copy, "F": mean_F_copy, "JF": mean_JF_copy}
        all_J_copy.append(mean_J_copy)
        all_F_copy.append(mean_F_copy)

    # Calculate overall means
    overall_J_flow = np.mean(all_J_flow)
    overall_F_flow = np.mean(all_F_flow)
    overall_JF_flow = (overall_J_flow + overall_F_flow) / 2

    overall_J_copy = np.mean(all_J_copy)
    overall_F_copy = np.mean(all_F_copy)
    overall_JF_copy = (overall_J_copy + overall_F_copy) / 2

    # Save baseline_optflow.json results
    output_dir = Path("eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "baseline_optflow.json", "w") as f:
        json.dump({
            "optical_flow": {
                "J_mean": overall_J_flow,
                "F_mean": overall_F_flow,
                "JF_mean": overall_JF_flow,
                "per_seq": results_flow
            },
            "copy_mask": {
                "J_mean": overall_J_copy,
                "F_mean": overall_F_copy,
                "JF_mean": overall_JF_copy,
                "per_seq": results_copy
            }
        }, f, indent=4)

    # Print baseline summary table
    print("\n" + "="*50)
    print("DAVIS-2017 BASELINES SUMMARY")
    print("="*50)
    print(f"{'Method':<20} | {'J Mean':<8} | {'F Mean':<8} | {'J&F Mean':<8}")
    print("-"*50)
    print(f"{'Copy Previous Mask':<20} | {overall_J_copy:<8.4f} | {overall_F_copy:<8.4f} | {overall_JF_copy:<8.4f}")
    print(f"{'Farneback OptFlow':<20} | {overall_J_flow:<8.4f} | {overall_F_flow:<8.4f} | {overall_JF_flow:<8.4f}")
    print("="*50 + "\n")
    print(f"Results saved to {output_dir / 'baseline_optflow.json'}")


if __name__ == "__main__":
    main()
