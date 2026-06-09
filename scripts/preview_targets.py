"""Visualize a few SAM3 soft targets to confirm they are object-shaped."""
import sys, os, torch
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

targets_dir = Path("data/sam3_targets/train2017")
pt_files = sorted(targets_dir.glob("ann_*.pt"))[:6]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, f in enumerate(pt_files):
    data = torch.load(f, map_location="cpu", weights_only=False)
    logits = data["soft_logits"].float().squeeze()  # [256, 256]
    probs = logits.sigmoid()
    mask_binary = (probs > 0.5).numpy().astype(np.uint8)
    
    # Stats
    fg_pct = mask_binary.sum() / mask_binary.size * 100
    
    axes[i].imshow(probs.numpy(), cmap='hot', vmin=0, vmax=1)
    axes[i].set_title(f"{f.stem}\nFG={fg_pct:.1f}%  box={data['boxes'].tolist()}")
    axes[i].axis('off')

plt.suptitle("SAM3 Soft Targets (sigmoid probabilities)\nShould show OBJECT shapes, not rectangles", fontsize=14)
plt.tight_layout()
plt.savefig("scripts/sam3_targets_preview.png", dpi=100)
print("Saved: scripts/sam3_targets_preview.png")

# Also show the binary mask shape analysis
for f in pt_files[:3]:
    data = torch.load(f, map_location="cpu", weights_only=False)
    logits = data["soft_logits"].float().squeeze()
    probs = logits.sigmoid()
    mask = (probs > 0.5).numpy().astype(np.uint8)
    
    # Check if it's roughly rectangular
    rows = mask.any(axis=1)
    cols = mask.any(axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]] if rows.any() else (0, 0)
    c_min, c_max = np.where(cols)[0][[0, -1]] if cols.any() else (0, 0)
    bbox_area = (r_max - r_min + 1) * (c_max - c_min + 1)
    mask_area = mask.sum()
    fill_ratio = mask_area / max(bbox_area, 1)
    
    print(f"{f.stem}: mask_area={mask_area}, bbox_area={bbox_area}, "
          f"fill_ratio={fill_ratio:.3f} ({'RECTANGULAR' if fill_ratio > 0.9 else 'object-shaped'})")
