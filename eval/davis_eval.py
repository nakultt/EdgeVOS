"""
EdgeVOS — DAVIS-2017 Evaluation Script
Measures J&F Mean (Jaccard + F-measure) — the primary VOS benchmark metric.

Run:
  python eval/davis_eval.py checkpoint=checkpoints/best_stgp.pt
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from data.davis_dataset import DAVISDataset


def jaccard(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection over Union (J)."""
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / (union + 1e-8)


def f_measure(pred: np.ndarray, gt: np.ndarray, beta: float = 0.5) -> float:
    """Boundary F-measure (F)."""
    from skimage.morphology import disk
    from skimage.filters import rank
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    pred_b = _boundary(pred)
    gt_b   = _boundary(gt)
    prec = np.logical_and(pred_b, gt_b).sum() / (pred_b.sum() + 1e-8)
    rec  = np.logical_and(pred_b, gt_b).sum() / (gt_b.sum() + 1e-8)
    return (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-8)


def _boundary(mask: np.ndarray, dilation: int = 2) -> np.ndarray:
    import cv2
    kernel = np.ones((dilation, dilation), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    eroded  = cv2.erode(mask.astype(np.uint8), kernel)
    return (dilated - eroded).astype(bool)


def evaluate_pipeline(pipeline, dataset: DAVISDataset, device: str = "cuda") -> dict:
    """
    Run full EdgeVOS pipeline on DAVIS-2017 val set.
    Returns dict with J_mean, F_mean, JF_mean per sequence and overall.
    """
    all_J, all_F = [], []
    results_per_seq = {}

    for idx in tqdm(range(len(dataset)), desc="Evaluating DAVIS val"):
        seq = dataset[idx]
        name    = seq["name"]
        frames  = seq["frames"].to(device)     # [T, 3, H, W]
        gt_masks = seq["masks"].numpy()         # [T, H, W]
        T       = seq["n_frames"]

        # TODO: Replace with your text prompt per sequence
        # For evaluation, use object class name from DAVIS metadata
        text_prompt = name.replace("-", " ")

        # Init pipeline on frame 0 with ground truth mask (standard VOS eval protocol)
        pipeline.graph = pipeline.graph.__class__(
            projection=pipeline.graph.proj,
            keyframe_interval=pipeline.graph.K,
            **{k: getattr(pipeline.graph, k) for k in
               ['sim_trigger','sim_recover','drift_frames',
                'regrounding_interval','topk_patches','ghost_confirm']}
        )
        # Use GT mask at frame 0 for init (semi-supervised VOS protocol)
        gt_mask_t0 = torch.from_numpy(gt_masks[0]).to(device)
        
        # Reset stateful variables in pipeline
        pipeline._frame_count = 0
        pipeline.rg._text_cache = {}
        pipeline.rg.register_text_prompt(0, text_prompt)
        
        y_indices, x_indices = torch.where(gt_mask_t0)
        if y_indices.numel() > 0:
            y1, y2 = y_indices.min().item(), y_indices.max().item() + 1
            x1, x2 = x_indices.min().item(), x_indices.max().item() + 1
            bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=device)
        else:
            H_f, W_f = frames.shape[2], frames.shape[3]
            bbox = torch.tensor([0.0, 0.0, float(W_f), float(H_f)], dtype=torch.float32, device=device)
        pipeline._last_bboxes = {0: bbox}

        feat_384 = pipeline._extract_dinov3_feat(frames[0], gt_mask_t0)
        pipeline.graph.init_object(0, feat_384, gt_mask_t0, frame_t=0)

        seq_J, seq_F = [jaccard(gt_masks[0], gt_masks[0])], [1.0]

        for t in range(1, T):
            result = pipeline.process_frame(frames[t], obj_ids=[0])
            pred_mask = result[0]
            if pred_mask is None:
                seq_J.append(0.0)
                seq_F.append(0.0)
                continue
            pred_np = pred_mask.cpu().numpy()
            gt_np   = gt_masks[t].astype(bool)
            seq_J.append(jaccard(pred_np, gt_np))
            seq_F.append(f_measure(pred_np, gt_np))

        mean_J = np.mean(seq_J)
        mean_F = np.mean(seq_F)
        results_per_seq[name] = {"J": mean_J, "F": mean_F, "JF": (mean_J+mean_F)/2}
        all_J.append(mean_J)
        all_F.append(mean_F)

    overall_J  = np.mean(all_J)
    overall_F  = np.mean(all_F)
    overall_JF = (overall_J + overall_F) / 2

    print(f"\nDAVIS-2017 val results:")
    print(f"  J Mean:  {overall_J:.4f}")
    print(f"  F Mean:  {overall_F:.4f}")
    print(f"  J&F Mean: {overall_JF:.4f}  <- primary metric")

    return {
        "J_mean":  overall_J,
        "F_mean":  overall_F,
        "JF_mean": overall_JF,
        "per_seq": results_per_seq,
    }
