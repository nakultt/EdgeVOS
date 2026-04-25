"""
EdgeVOS -- Pre-cache SAM 3.1 soft mask logits using COCO bounding box prompts.

Instead of automatic/text mode (which produced only 36/9999 targets),
this script tells SAM 3.1 EXACTLY what to segment using COCO ground-truth
bounding boxes as prompts. This guarantees clean, high-quality soft targets.

Output per annotation:
    {
        "soft_logits": Tensor [1, 256, 256] FP16,   # pre-sigmoid mask logits
        "boxes":       Tensor [4]           FP32,    # xyxy absolute pixels
        "ann_id":      int,
        "category_id": int,
    }

Usage
-----
    python scripts/cache_sam3_targets.py \
        --coco_root data/COCO \
        --output_dir data/sam3_targets \
        --split train2017

Resume-safe: existing .pt files are skipped automatically.

Expected output: ~25,000-28,000 targets from val2017 (5000 images, ~36k annotations,
filtered for area >= 400px and no crowd annotations).
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cache SAM 3.1 soft mask logits using COCO bbox prompts.",
    )
    p.add_argument("--coco_root",   type=str, default="data/COCO")
    p.add_argument("--output_dir",  type=str, default="data/sam3_targets")
    p.add_argument("--split",       type=str, default="val2017",
                   help="COCO split: val2017 (1GB) or train2017 (18GB)")
    p.add_argument("--min_area",    type=int, default=400,
                   help="Skip annotations with area < this (roughly 20x20px)")
    p.add_argument("--min_box_dim", type=int, default=10,
                   help="Skip boxes whose width or height < this in pixels")
    p.add_argument("--mask_size",   type=int, default=256,
                   help="Spatial size of saved soft logits")
    p.add_argument("--device",      type=str, default="cuda")
    p.add_argument("--max_anns",    type=int, default=None,
                   help="Optional cap on annotations (for quick testing)")
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────

def gpu_mem_mb() -> float:
    return torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0


def build_sam3(device: str):
    """Build the SAM 3.1 model + Sam3Processor for box-prompted inference.

    Uses Sam3Processor.add_geometric_prompt() which accepts boxes in
    [cx, cy, w, h] normalised [0,1] format and runs grounding inference.
    """
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("[cache] Building SAM 3.1 model ...")
    model = build_sam3_image_model(device=device, eval_mode=True)
    # Set confidence threshold very low so box-prompted results aren't filtered
    processor = Sam3Processor(model=model, device=device, confidence_threshold=0.01)
    print(f"[cache] SAM 3.1 ready | GPU mem: {gpu_mem_mb():.0f} MB")
    return model, processor


def coco_to_xyxy(bbox_xywh) -> list:
    """Convert COCO [x, y, w, h] → [x1, y1, x2, y2] absolute pixels."""
    x, y, w, h = bbox_xywh
    return [x, y, x + w, y + h]


def xyxy_to_cxcywh_norm(bbox_xyxy, img_w, img_h) -> list:
    """Convert [x1, y1, x2, y2] absolute → [cx, cy, w, h] normalised [0,1].

    This is the format Sam3Processor.add_geometric_prompt() expects.
    """
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return [cx, cy, w, h]


def load_annotations(coco_root: str, split: str):
    """Load COCO annotations. Returns pycocotools COCO object."""
    from pycocotools.coco import COCO

    ann_file = Path(coco_root) / "annotations" / f"instances_{split}.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotations not found: {ann_file}")

    print(f"[cache] Loading {ann_file} ...")
    coco = COCO(str(ann_file))
    return coco


# ── Core extraction ──────────────────────────────────────────────────

def extract_target_from_box(
    processor,
    image:     Image.Image,
    bbox_xyxy: list,
    mask_size: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Run SAM 3.1 with a GROUND-TRUTH bounding box prompt.

    Uses Sam3Processor API:
      1. processor.set_image(image)  — encode backbone features
      2. processor.add_geometric_prompt(box_cxcywh_norm, label=True, state)
         — run grounding with the box as a positive geometric prompt

    The masks_logits in the state are post-sigmoid probabilities at original
    image resolution.  We recover pre-sigmoid logits via torch.logit() and
    resize to mask_size (256) for distillation.

    Parameters
    ----------
    processor  : Sam3Processor
    image      : PIL Image RGB
    bbox_xyxy  : [x1, y1, x2, y2] absolute pixel coordinates
    mask_size  : output spatial resolution (256)

    Returns
    -------
    dict: soft_logits [1, 256, 256] FP16 + boxes [4] FP32
    or None if SAM returns empty output
    """
    img_w, img_h = image.size

    # 1. Encode image
    state = processor.set_image(image)

    # 2. Convert box to [cx, cy, w, h] normalised [0,1]
    box_norm = xyxy_to_cxcywh_norm(bbox_xyxy, img_w, img_h)

    # 3. Run box-prompted grounding
    state = processor.add_geometric_prompt(
        box=box_norm, label=True, state=state
    )

    # 4. Check outputs
    if "masks_logits" not in state or state["masks_logits"] is None:
        processor.reset_all_prompts(state)
        return None

    mask_probs = state["masks_logits"]  # [N, 1, H, W] post-sigmoid probabilities
    if mask_probs.numel() == 0:
        processor.reset_all_prompts(state)
        return None

    # Pick the first (best) mask
    mask_prob = mask_probs[0]  # [1, H, W]

    # Check mask is non-degenerate
    n_above = (mask_prob > 0.5).sum().item()
    if n_above < 50:
        processor.reset_all_prompts(state)
        return None

    # Recover pre-sigmoid logits for MSE distillation loss
    logits = torch.logit(mask_prob.float().clamp(1e-6, 1.0 - 1e-6))

    # Resize to target mask_size
    logits_resized = F.interpolate(
        logits.unsqueeze(0),         # [1, 1, H, W]
        size=(mask_size, mask_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)                      # [1, mask_size, mask_size]

    # Reset prompts so next call starts fresh
    processor.reset_all_prompts(state)

    return {
        "soft_logits": logits_resized.half().cpu(),   # [1, 256, 256] FP16
        "boxes":       torch.tensor(bbox_xyxy, dtype=torch.float32),  # [4] xyxy
    }


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    out_dir  = Path(args.output_dir)
    img_dir  = Path(args.coco_root) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {img_dir}\n"
            f"Download: wget http://images.cocodataset.org/zips/{args.split}.zip"
        )

    # Load COCO
    coco = load_annotations(args.coco_root, args.split)

    # Pre-scan: which images actually exist on disk?
    print("[cache] Scanning available images ...")
    available_img_ids = set()
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        if (img_dir / img_info["file_name"]).exists():
            available_img_ids.add(img_id)
    print(f"[cache] Images on disk: {len(available_img_ids)} / {len(coco.getImgIds())}")

    # Get all annotation IDs and filter
    print("[cache] Filtering annotations ...")
    all_ann_ids = coco.getAnnIds()
    all_anns    = coco.loadAnns(all_ann_ids)

    valid = []
    for ann in all_anns:
        if ann.get("iscrowd", 0):
            continue
        if ann["area"] < args.min_area:
            continue
        bx, by, bw, bh = ann["bbox"]
        if bw < args.min_box_dim or bh < args.min_box_dim:
            continue
        # Only keep annotations whose images exist
        if ann["image_id"] not in available_img_ids:
            continue
        valid.append(ann)

    # Optional cap for testing
    if args.max_anns is not None:
        valid = valid[:args.max_anns]

    # Resume support: skip annotations already cached
    already_cached = 0
    todo = []
    for ann in valid:
        pt_path = out_dir / f"ann_{ann['id']}.pt"
        if pt_path.exists():
            already_cached += 1
        else:
            todo.append(ann)

    print(f"[cache] Total valid annotations : {len(valid)}")
    print(f"[cache] Already cached (skipped): {already_cached}")
    print(f"[cache] To process              : {len(todo)}")
    print(f"[cache] Output                  : {out_dir}")

    if not todo:
        print("[cache] Nothing to do — all annotations cached. Exiting.")
        return

    # Build SAM 3.1
    model, processor = build_sam3(device)

    # Preload image cache (avoid re-opening same image for multiple annotations)
    current_img_id   = None
    current_img_pil  = None

    saved = 0
    skipped_empty = 0
    errors = 0
    t_start = time.time()

    for i, ann in enumerate(tqdm(todo, desc="Caching SAM 3.1 targets (bbox mode)")):
        ann_id   = ann["id"]
        img_id   = ann["image_id"]
        cat_id   = ann["category_id"]
        bbox_xyxy = coco_to_xyxy(ann["bbox"])

        # Load image (reuse if same image as last annotation — saves disk I/O)
        if img_id != current_img_id:
            img_info = coco.loadImgs(img_id)[0]
            img_path = img_dir / img_info["file_name"]
            try:
                current_img_pil = Image.open(img_path).convert("RGB")
                current_img_id  = img_id
            except Exception as e:
                tqdm.write(f"[WARN] Cannot open {img_path}: {e}")
                errors += 1
                continue

        # Run SAM 3.1 with bounding box prompt
        try:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
                target = extract_target_from_box(
                    processor, current_img_pil, bbox_xyxy, args.mask_size
                )
        except Exception as e:
            tqdm.write(f"[WARN] SAM inference error on ann {ann_id}: {e}")
            errors += 1
            continue

        if target is None:
            skipped_empty += 1
            continue

        # Add metadata
        target["ann_id"]     = ann_id
        target["category_id"] = cat_id

        torch.save(target, out_dir / f"ann_{ann_id}.pt")
        saved += 1

        # Log + memory cleanup every 500 annotations
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            speed   = (i + 1) / max(elapsed, 1e-6)
            tqdm.write(
                f"[cache] {i+1}/{len(todo)} | saved={saved} "
                f"empty={skipped_empty} errs={errors} | "
                f"GPU: {gpu_mem_mb():.0f} MB | {speed:.1f} ann/s"
            )
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    elapsed = time.time() - t_start
    print(
        f"\n[cache] Done in {elapsed/60:.1f} min\n"
        f"  Valid annotations processed : {len(todo)}\n"
        f"  Saved targets               : {saved}\n"
        f"  Empty/degenerate skipped    : {skipped_empty}\n"
        f"  Errors                      : {errors}\n"
        f"  Output dir                  : {out_dir.resolve()}\n"
        f"  Avg speed                   : {len(todo)/max(elapsed,1):.1f} ann/s\n"
        f"\n  Expected for val2017: ~25,000-28,000 targets\n"
        f"  If saved << expected: check that images exist in {img_dir}"
    )


if __name__ == "__main__":
    main()