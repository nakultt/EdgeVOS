"""
EdgeVOS -- Pre-cache SAM 3.1 soft mask logits for COCO train2017.

These cached targets serve as distillation supervision for the
MLP mask head (Module 2).  Each output file contains:

    {
        "soft_logits": Tensor [1, 256, 256] FP16,   # pre-sigmoid mask logits
        "boxes":       Tensor [4]           FP32,    # xyxy absolute pixels
    }

Usage
-----
    python scripts/cache_sam3_targets.py \
        --coco_root data/COCO \
        --output_dir data/sam3_targets

Resume-safe: existing .pt files are skipped automatically.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cache SAM 3.1 soft mask logits for COCO distillation.",
    )
    p.add_argument(
        "--coco_root", type=str, default="data/COCO",
        help="Path to COCO dataset root (expects train2017/ and annotations/).",
    )
    p.add_argument(
        "--output_dir", type=str, default="data/sam3_targets",
        help="Directory for cached .pt files.",
    )
    p.add_argument(
        "--num_images", type=int, default=10_000,
        help="Number of images to process (10%% subset of COCO train2017).",
    )
    p.add_argument(
        "--batch_size", type=int, default=4,
        help="Images processed per batch (memory management granularity).",
    )
    p.add_argument(
        "--confidence", type=float, default=0.7,
        help="Minimum SAM 3.1 confidence to keep a mask proposal.",
    )
    p.add_argument(
        "--max_proposals", type=int, default=3,
        help="Maximum mask proposals to consider per image.",
    )
    p.add_argument(
        "--mask_size", type=int, default=256,
        help="Spatial size of saved soft logits.",
    )
    p.add_argument(
        "--device", type=str, default="cuda",
        help="Inference device.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Seed for reproducible image subset selection.",
    )
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────

def gpu_mem_mb() -> float:
    """Current GPU memory allocated in MB (0 if no CUDA)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def load_coco_image_ids(
    coco_root: str,
    num_images: int,
    seed: int,
) -> List[dict]:
    """
    Load COCO train2017 image metadata and return a reproducible
    random subset of *num_images* entries.

    Returns
    -------
    List of dicts from pycocotools, each with 'id', 'file_name', etc.
    """
    from pycocotools.coco import COCO

    ann_path = os.path.join(coco_root, "annotations", "instances_train2017.json")
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(
            f"COCO annotations not found at {ann_path}. "
            f"Ensure --coco_root points to the COCO root directory."
        )

    print(f"[cache] Loading COCO annotations from {ann_path} ...")
    coco = COCO(ann_path)
    all_ids = sorted(coco.getImgIds())

    rng = random.Random(seed)
    subset_ids = rng.sample(all_ids, min(num_images, len(all_ids)))
    img_infos = coco.loadImgs(subset_ids)

    print(f"[cache] Selected {len(img_infos)} / {len(all_ids)} images (seed={seed})")
    return img_infos


def build_sam3(device: str):
    """
    Build the SAM 3 image model and processor.

    Returns
    -------
    (model, processor)
    """
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("[cache] Building SAM 3 model (downloading checkpoint if needed) ...")
    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        enable_segmentation=True,
    )

    processor = Sam3Processor(
        model=model,
        resolution=1008,
        device=device,
        confidence_threshold=0.3,  # low initial threshold; we filter at --confidence
    )

    print(f"[cache] SAM 3 ready on {device} | GPU mem: {gpu_mem_mb():.0f} MB")
    return model, processor


# ── Core extraction ──────────────────────────────────────────────────

def extract_target(
    processor,
    image: Image.Image,
    confidence: float,
    max_proposals: int,
    mask_size: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Run SAM 3 on a single image and return the best mask as a
    distillation target dict, or None if no proposal passes the
    confidence threshold.

    The processor's ``_forward_grounding`` applies sigmoid to the raw
    mask logits and stores the result in ``state["masks_logits"]``.
    We invert this with ``torch.logit`` to recover pre-sigmoid values
    suitable for MSE-based distillation loss.

    Parameters
    ----------
    processor : Sam3Processor
    image : PIL Image (RGB)
    confidence : float  -- minimum score to accept
    max_proposals : int -- max proposals to consider
    mask_size : int     -- spatial size of saved logits (256)

    Returns
    -------
    dict with 'soft_logits' [1, mask_size, mask_size] FP16
              and 'boxes' [4] FP32 xyxy absolute
    or None
    """
    # 1. Encode image
    state = processor.set_image(image)

    # 2. Run grounding with generic prompt for automatic proposals
    state = processor.set_text_prompt(prompt="object", state=state)

    # 3. Check outputs
    if "scores" not in state or len(state["scores"]) == 0:
        return None

    scores = state["scores"]          # [N]
    masks_soft = state["masks_logits"]  # [N, 1, H, W] post-sigmoid
    boxes = state["boxes"]             # [N, 4] xyxy absolute

    # 4. Confidence filter
    keep = scores > confidence
    if not keep.any():
        return None

    scores = scores[keep]
    masks_soft = masks_soft[keep]
    boxes = boxes[keep]

    # 5. Top-k by score, pick the best
    k = min(max_proposals, len(scores))
    best_idx = scores.topk(k).indices[0]

    best_mask_soft = masks_soft[best_idx]          # [1, H, W] values in (0, 1)
    best_box = boxes[best_idx]                     # [4]

    # 6. Invert sigmoid -> raw logits (clamp for numerical stability)
    best_logits = torch.logit(best_mask_soft.clamp(1e-6, 1.0 - 1e-6))

    # 7. Resize to target mask size
    best_logits = F.interpolate(
        best_logits.unsqueeze(0),          # [1, 1, H, W]
        size=(mask_size, mask_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)                            # [1, mask_size, mask_size]

    return {
        "soft_logits": best_logits.half().cpu(),   # [1, 256, 256] FP16
        "boxes": best_box.float().cpu(),           # [4] xyxy absolute
    }


# ── Main loop ────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO metadata
    img_infos = load_coco_image_ids(args.coco_root, args.num_images, args.seed)
    img_dir = Path(args.coco_root) / "train2017"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    # Filter out already-cached images (resume safety)
    todo = [info for info in img_infos if not (out_dir / f"{info['id']}.pt").exists()]
    skipped = len(img_infos) - len(todo)
    if skipped:
        print(f"[cache] Skipping {skipped} already-cached images")
    print(f"[cache] Processing {len(todo)} images -> {out_dir}")

    if len(todo) == 0:
        print("[cache] Nothing to do. Exiting.")
        return

    # Build model
    model, processor = build_sam3(device)

    # Counters
    saved_count = 0
    low_conf_count = 0
    error_count = 0
    t_start = time.time()

    # Process in batches
    pbar = tqdm(range(0, len(todo), args.batch_size), desc="Caching SAM 3 targets")

    for batch_start in pbar:
        batch = todo[batch_start : batch_start + args.batch_size]

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
            for info in batch:
                img_id = info["id"]
                img_path = img_dir / info["file_name"]

                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as e:
                    tqdm.write(f"[WARN] Failed to load {img_path}: {e}")
                    error_count += 1
                    continue

                target = extract_target(
                    processor,
                    image,
                    confidence=args.confidence,
                    max_proposals=args.max_proposals,
                    mask_size=args.mask_size,
                )

                if target is None:
                    low_conf_count += 1
                    continue

                # Save
                save_path = out_dir / f"{img_id}.pt"
                torch.save(target, save_path)
                saved_count += 1

                # Reset processor state to free GPU memory
                processor.reset_all_prompts(
                    processor.__dict__.get("_last_state", {})
                )

        # Memory management between batches
        if device == "cuda":
            torch.cuda.empty_cache()

        # Progress update
        total_processed = batch_start + len(batch)
        pbar.set_postfix(saved=saved_count, low_conf=low_conf_count, errs=error_count)

        # Memory log every 100 images
        if total_processed % 100 < args.batch_size:
            elapsed = time.time() - t_start
            speed = total_processed / max(elapsed, 1e-6)
            tqdm.write(
                f"[cache] {total_processed}/{len(todo)} | "
                f"saved={saved_count} low_conf={low_conf_count} errs={error_count} | "
                f"GPU mem: {gpu_mem_mb():.0f} MB | "
                f"speed: {speed:.1f} img/s"
            )

        gc.collect()

    # Final summary
    elapsed = time.time() - t_start
    print(
        f"\n[cache] Done in {elapsed / 60:.1f} min\n"
        f"  Total processed : {len(todo)}\n"
        f"  Saved           : {saved_count}\n"
        f"  Low confidence  : {low_conf_count}\n"
        f"  Errors          : {error_count}\n"
        f"  Output dir      : {out_dir.resolve()}\n"
        f"  Avg speed       : {len(todo) / max(elapsed, 1):.1f} img/s"
    )


if __name__ == "__main__":
    main()
