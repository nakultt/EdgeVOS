# scripts/test_checkpoint.py
"""Quick visual test on a distillation checkpoint."""
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import sys
sys.path.insert(0, '.')
import argparse

from omegaconf import OmegaConf
from data.davis_dataset import COCODistillDataset
from train.distill_mask import StudentModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-index", type=int, default=100)
    parser.add_argument("--output", type=str, default="test_output.jpg")
    return parser.parse_args()

args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load checkpoint
ckpt = torch.load('checkpoints/best_distill.pt', map_location=device)
print(f"Checkpoint epoch : {ckpt['epoch']}")
print(f"Checkpoint loss  : {ckpt['loss']:.4f}")

cfg = OmegaConf.create(ckpt["cfg"])
cfg.model.backbone_pretrained = False
model = StudentModel(cfg).to(device)
model.load_state_dict(ckpt['model'])
model.eval()
print("Model loaded successfully")

# Pick a cached target
targets = sorted(Path('data/sam3_targets/train2017').glob('*.pt'))
target_path = targets[args.target_index]
dataset = COCODistillDataset(
    coco_root=cfg.data.coco_root,
    targets_dir=cfg.data.sam3_targets_dir,
    img_size=cfg.data.img_size,
)
sample = dataset[args.target_index]

print(f"Target file          : {target_path}")
print(f"\nTeacher logit range : {sample['soft_logits'].float().min():.2f} / {sample['soft_logits'].float().max():.2f}")

img_tensor = sample["image"].unsqueeze(0).to(device)

# Forward pass
with torch.no_grad():
    pred_logits = model(img_tensor)  # [1, 1, 256, 256]

pred_mask    = pred_logits.sigmoid().squeeze().cpu().numpy()
teacher_mask = sample['soft_logits'].float().sigmoid().squeeze().numpy()

print(f"\nPred   mask — min/max : {pred_mask.min():.3f} / {pred_mask.max():.3f}")
print(f"Teacher mask — min/max: {teacher_mask.min():.3f} / {teacher_mask.max():.3f}")

# IoU between student and teacher, binarised at 0.5.
pred_bin    = pred_mask > 0.5
teacher_bin = teacher_mask > 0.5
inter = (pred_bin & teacher_bin).sum()
union = (pred_bin | teacher_bin).sum()
iou   = inter / (union + 1e-6)
print(f"Student vs Teacher IoU : {iou:.3f}  (target after 30 epochs: >0.90)")

# Save visual comparison
crop_rgb = (sample["image"].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
orig = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
H, W = orig.shape[:2]

pred_vis    = cv2.resize((pred_mask    * 255).astype(np.uint8), (W, H))
teacher_vis = cv2.resize((teacher_mask * 255).astype(np.uint8), (W, H))

pred_color    = cv2.applyColorMap(pred_vis,    cv2.COLORMAP_JET)
teacher_color = cv2.applyColorMap(teacher_vis, cv2.COLORMAP_JET)

# Resize all panels to same height for hstack
panels = [orig, teacher_color, pred_color]
panels = [cv2.resize(p, (W, H)) for p in panels]
combined = np.hstack(panels)

cv2.imwrite(args.output, combined)
print(f"\nSaved: {args.output}")
print("Left=object crop | Middle=SAM3.1 teacher crop | Right=student crop")
