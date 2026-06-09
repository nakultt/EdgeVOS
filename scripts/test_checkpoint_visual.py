"""
Test the existing best_distill.pt (SpatialMaskHead) checkpoint on DAVIS images.

For each test image:
  1. Load the original frame + GT annotation from DAVIS
  2. Find the object bounding box from the GT mask
  3. Crop the object region (with padding), resize to 224x224
  4. Run through backbone._extract_feature_map() -> SpatialMaskHead
  5. Paste the predicted mask back onto the full frame
  6. Save a side-by-side comparison: Original | GT Mask | Predicted Mask

Output: scripts/checkpoint_test_results.png
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.backbone import BackboneWithRoI
from models.mask_head import SpatialMaskHead


def load_model(ckpt_path, device):
    """Load the SpatialMaskHead student model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint: epoch={ckpt['epoch']}, val_iou={ckpt['val_iou']:.4f}, "
          f"head_type={ckpt.get('mask_head_type', 'unknown')}")

    backbone = BackboneWithRoI(model_name="repvit_m1", pretrained=False).to(device)
    mask_head = SpatialMaskHead(
        in_channels=backbone.feat_channels,  # 384
        out_size=256,
    ).to(device)

    # Load weights - filter backbone and mask_head keys
    state = ckpt["model"]
    bb_state = {k.replace("backbone.", "", 1): v for k, v in state.items() if k.startswith("backbone.")}
    mh_state = {k.replace("mask_head.", "", 1): v for k, v in state.items() if k.startswith("mask_head.")}

    backbone.load_state_dict(bb_state)
    mask_head.load_state_dict(mh_state)

    backbone.eval()
    mask_head.eval()
    return backbone, mask_head


def expand_box(box, img_w, img_h, pad_ratio=0.15):
    """Expand a bounding box by pad_ratio."""
    x1, y1, x2, y2 = box
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    pad = max(w, h) * pad_ratio
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(img_w, x2 + pad),
        min(img_h, y2 + pad),
    ]


def predict_mask(backbone, mask_head, img_pil, bbox, device):
    """
    Crop the image to bbox, run through model, return mask at crop size.
    
    Args:
        img_pil: PIL image (full frame)
        bbox: [x1, y1, x2, y2] absolute pixel coords
    Returns:
        crop_mask: numpy bool array at crop resolution
        crop_img: numpy array of the cropped region
    """
    img_w, img_h = img_pil.size
    ex1, ey1, ex2, ey2 = expand_box(bbox, img_w, img_h, pad_ratio=0.15)
    ex1, ey1 = int(ex1), int(ey1)
    ex2, ey2 = int(ex2), int(ey2)

    # Crop and resize to 224x224
    crop = img_pil.crop((ex1, ey1, ex2, ey2))
    crop_resized = crop.resize((224, 224), Image.BILINEAR)
    crop_tensor = torch.from_numpy(np.array(crop_resized)).permute(2, 0, 1).float() / 255.0
    crop_tensor = crop_tensor.unsqueeze(0).to(device)  # [1, 3, 224, 224]

    with torch.no_grad():
        feat_map = backbone._extract_feature_map(crop_tensor)  # [1, 384, 14, 14]
        logits = mask_head(feat_map)                            # [1, 1, 256, 256]
        probs = logits.sigmoid().squeeze().cpu().numpy()        # [256, 256]
        mask_binary = probs > 0.5

    return mask_binary, probs, np.array(crop_resized), (ex1, ey1, ex2, ey2)


def get_bbox_from_mask(mask_np):
    """Get bounding box [x1, y1, x2, y2] from a binary mask."""
    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [x1, y1, x2 + 1, y2 + 1]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    backbone, mask_head = load_model("checkpoints/best_distill.pt", device)

    # Test sequences from DAVIS
    davis_root = Path("data/DAVIS")
    test_sequences = ["blackswan", "car-shadow", "dog", "bear"]
    # Filter to sequences that exist
    test_sequences = [s for s in test_sequences if (davis_root / "JPEGImages" / "480p" / s).exists()]
    print(f"Testing on {len(test_sequences)} sequences: {test_sequences}")

    if not test_sequences:
        print("ERROR: No DAVIS sequences found!")
        return

    # Create figure: 1 row per sequence, 4 columns
    n_seq = len(test_sequences)
    fig, axes = plt.subplots(n_seq, 4, figsize=(20, 5 * n_seq))
    if n_seq == 1:
        axes = axes[np.newaxis, :]

    for row, seq_name in enumerate(test_sequences):
        img_dir = davis_root / "JPEGImages" / "480p" / seq_name
        ann_dir = davis_root / "Annotations" / "480p" / seq_name

        # Use first frame
        img_path = sorted(img_dir.glob("*.jpg"))[0]
        ann_path = sorted(ann_dir.glob("*.png"))[0]

        img_pil = Image.open(img_path).convert("RGB")
        ann_pil = Image.open(ann_path)
        gt_mask = np.array(ann_pil)
        gt_binary = (gt_mask > 0).astype(np.uint8)

        # Get bbox from GT mask
        bbox = get_bbox_from_mask(gt_binary)
        if bbox is None:
            print(f"  {seq_name}: No object in GT mask, skipping")
            continue

        # Run prediction
        pred_mask, pred_probs, crop_img, expanded_box = predict_mask(
            backbone, mask_head, img_pil, bbox, device
        )
        ex1, ey1, ex2, ey2 = expanded_box

        # Paste predicted mask back to full frame
        img_np = np.array(img_pil)
        H, W = img_np.shape[:2]
        full_pred = np.zeros((H, W), dtype=bool)
        crop_h = ey2 - ey1
        crop_w = ex2 - ex1
        resized_mask = np.array(
            Image.fromarray(pred_mask.astype(np.uint8) * 255).resize(
                (crop_w, crop_h), Image.NEAREST
            )
        ) > 127
        full_pred[ey1:ey2, ex1:ex2] = resized_mask

        # Compute IoU
        intersection = (full_pred & gt_binary.astype(bool)).sum()
        union = (full_pred | gt_binary.astype(bool)).sum()
        iou = intersection / max(union, 1)

        # FG pixel stats
        fg_pct = pred_mask.sum() / pred_mask.size * 100
        fill_ratio = 0
        if pred_mask.any():
            rows = pred_mask.any(axis=1)
            cols = pred_mask.any(axis=0)
            r_min, r_max = np.where(rows)[0][[0, -1]]
            c_min, c_max = np.where(cols)[0][[0, -1]]
            bbox_area = max((r_max - r_min + 1) * (c_max - c_min + 1), 1)
            fill_ratio = pred_mask.sum() / bbox_area

        print(f"  {seq_name}: IoU={iou:.3f}, FG={fg_pct:.1f}%, "
              f"fill_ratio={fill_ratio:.3f} ({'RECT' if fill_ratio > 0.9 else 'object-shaped'})")

        # ── Plot ────────────────────────────────────────────────────
        # Col 0: Original image with bbox
        axes[row, 0].imshow(img_np)
        x1, y1, x2, y2 = bbox
        from matplotlib.patches import Rectangle
        rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='lime', facecolor='none')
        axes[row, 0].add_patch(rect)
        axes[row, 0].set_title(f"{seq_name} - Original + BBox", fontsize=11)
        axes[row, 0].axis("off")

        # Col 1: GT mask overlay
        overlay_gt = img_np.copy()
        overlay_gt[gt_binary > 0] = (
            overlay_gt[gt_binary > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        ).astype(np.uint8)
        axes[row, 1].imshow(overlay_gt)
        axes[row, 1].set_title("Ground Truth Mask", fontsize=11)
        axes[row, 1].axis("off")

        # Col 2: Predicted probability heatmap (crop)
        axes[row, 2].imshow(crop_img)
        axes[row, 2].imshow(pred_probs, cmap="hot", alpha=0.6, vmin=0, vmax=1)
        axes[row, 2].set_title(f"Predicted Probs (crop)\nFG={fg_pct:.1f}% fill={fill_ratio:.2f}", fontsize=11)
        axes[row, 2].axis("off")

        # Col 3: Predicted mask overlay on full frame
        overlay_pred = img_np.copy()
        overlay_pred[full_pred] = (
            overlay_pred[full_pred] * 0.5 + np.array([255, 50, 50]) * 0.5
        ).astype(np.uint8)
        axes[row, 3].imshow(overlay_pred)
        axes[row, 3].set_title(f"Predicted Mask (IoU={iou:.3f})", fontsize=11)
        axes[row, 3].axis("off")

    plt.suptitle(
        "SpatialMaskHead Checkpoint Test (best_distill.pt)\n"
        "Cols: Original+BBox | GT Mask (green) | Predicted Probs (crop) | Predicted Mask (red)",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    out_path = "scripts/checkpoint_test_results.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
