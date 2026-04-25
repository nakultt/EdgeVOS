import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="data/sam3_targets/train2017")
    parser.add_argument("--coco_root", type=str, default="data/COCO")
    parser.add_argument("--split", type=str, default="train2017")
    parser.add_argument("--output_dir", type=str, default="verification_results")
    parser.add_argument("--num_files", type=int, default=20)
    return parser.parse_args()

def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pt_files = list(cache_dir.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {cache_dir}")
        return
        
    pt_files = pt_files[:args.num_files]
    print(f"Found {len(pt_files)} files to verify.")

    ann_file = Path(args.coco_root) / "annotations" / f"instances_{args.split}.json"
    print(f"Loading COCO annotations from {ann_file} (this only happens ONCE!)...")
    coco = COCO(str(ann_file))

    for pt_path in tqdm(pt_files, desc="Verifying targets"):
        target = torch.load(pt_path, weights_only=False)
        ann_id = target.get("ann_id")
        
        if ann_id is None:
            continue

        ann_info = coco.loadAnns([ann_id])
        if not ann_info:
            continue
            
        img_id = ann_info[0]["image_id"]
        img_info = coco.loadImgs(img_id)[0]
        img_path = Path(args.coco_root) / args.split / img_info["file_name"]
        
        if not img_path.exists():
            continue
            
        img_pil = Image.open(img_path).convert("RGB")
        img_w, img_h = img_pil.size
        
        logits = target["soft_logits"].float()
        probs = torch.sigmoid(logits).squeeze().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Verification for Annotation ID: {ann_id} (Img: {img_info['file_name']})", fontsize=14)

        ax_mask = axes[0]
        im = ax_mask.imshow(probs, cmap="viridis", vmin=0, vmax=1)
        ax_mask.set_title("Soft Probabilities (256x256)")
        fig.colorbar(im, ax=ax_mask, fraction=0.046, pad=0.04)

        ax_img = axes[1]
        ax_img.imshow(img_pil)
        
        probs_resized = F.interpolate(
            logits.unsqueeze(0), size=(img_h, img_w), mode="bilinear", align_corners=False
        ).sigmoid().squeeze().numpy()
        
        ax_img.imshow(probs_resized, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        
        box = target["boxes"].numpy()
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, 
            linewidth=2, edgecolor='red', facecolor='none', label="Prompt Box"
        )
        ax_img.add_patch(rect)
        ax_img.legend()
        ax_img.set_title(f"Overlay on Original ({img_w}x{img_h})")

        plt.tight_layout()
        out_path = out_dir / f"verify_ann_{ann_id}.png"
        plt.savefig(out_path)
        plt.close(fig)

if __name__ == "__main__":
    main()
