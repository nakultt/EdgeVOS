import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Verify a cached SAM 3.1 target (.pt file).")
    parser.add_argument("pt_file", type=str, help="Path to the .pt file to verify")
    parser.add_argument("--coco_root", type=str, default="data/COCO", help="Path to COCO dataset")
    parser.add_argument("--split", type=str, default="train2017", help="COCO split (e.g., train2017)")
    parser.add_argument("--output", type=str, default="verify_output.png", help="Where to save the visualization")
    return parser.parse_args()

def main():
    args = parse_args()
    pt_path = Path(args.pt_file)
    
    if not pt_path.exists():
        print(f"Error: Could not find {pt_path}")
        return

    print(f"Loading {pt_path}...")
    target = torch.load(pt_path, weights_only=False)
    
    # 1. Print raw dictionary contents
    print("\n--- Target Contents ---")
    for key, value in target.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor of shape {list(value.shape)}, dtype {value.dtype}")
        else:
            print(f"  {key}: {value} (type: {type(value).__name__})")

    ann_id = target.get("ann_id")
    if ann_id is None:
        print("Error: 'ann_id' not found in the .pt file.")
        return

    # 2. Load COCO to find the original image
    ann_file = Path(args.coco_root) / "annotations" / f"instances_{args.split}.json"
    if not ann_file.exists():
        print(f"\nWarning: COCO annotations not found at {ann_file}.")
        print("Cannot overlay on original image. Will just plot the raw mask.")
        img_pil = None
    else:
        print(f"\nLoading COCO annotations from {ann_file} to find the original image...")
        coco = COCO(str(ann_file))
        ann_info = coco.loadAnns([ann_id])
        if not ann_info:
            print(f"Error: Annotation ID {ann_id} not found in {args.split}.")
            img_pil = None
        else:
            img_id = ann_info[0]["image_id"]
            img_info = coco.loadImgs(img_id)[0]
            img_path = Path(args.coco_root) / args.split / img_info["file_name"]
            
            if not img_path.exists():
                print(f"Error: Image not found at {img_path}")
                img_pil = None
            else:
                print(f"Found original image: {img_info['file_name']}")
                img_pil = Image.open(img_path).convert("RGB")

    # 3. Process the soft logits
    logits = target["soft_logits"].float()  # Convert FP16 to FP32
    # Apply sigmoid to convert raw logits back to probabilities [0.0 to 1.0]
    probs = torch.sigmoid(logits).squeeze().numpy()  # Shape: [256, 256]

    # 4. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Verification for Annotation ID: {ann_id}", fontsize=16)

    # Left plot: Raw 256x256 soft mask probabilities
    ax_mask = axes[0]
    im = ax_mask.imshow(probs, cmap="viridis", vmin=0, vmax=1)
    ax_mask.set_title("Soft Probabilities (256x256)")
    fig.colorbar(im, ax=ax_mask, fraction=0.046, pad=0.04)

    # Right plot: Overlay on original image
    ax_img = axes[1]
    if img_pil is not None:
        img_w, img_h = img_pil.size
        ax_img.imshow(img_pil)
        
        # Resize probabilities to match original image size for overlay
        probs_resized = F.interpolate(
            logits.unsqueeze(0), size=(img_h, img_w), mode="bilinear", align_corners=False
        ).sigmoid().squeeze().numpy()
        
        # Overlay the mask (alpha=0.5)
        ax_img.imshow(probs_resized, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        
        # Draw the bounding box
        box = target["boxes"].numpy()  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, 
            linewidth=2, edgecolor='red', facecolor='none', label="Prompt Box"
        )
        ax_img.add_patch(rect)
        ax_img.legend()
        ax_img.set_title(f"Overlay on Original ({img_w}x{img_h})")
    else:
        ax_img.text(0.5, 0.5, 'Original Image Not Found', horizontalalignment='center', verticalalignment='center')
        ax_img.axis('off')

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"\nSuccess! Visualization saved to {args.output}")

if __name__ == "__main__":
    main()
