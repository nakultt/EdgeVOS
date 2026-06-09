"""Quick inspection of a sam3 target .pt file to understand structure."""
import sys, os, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path

targets_dir = Path("data/sam3_targets/train2017")
pt_files = sorted(targets_dir.glob("ann_*.pt"))[:3]

for f in pt_files:
    print(f"\n{'='*60}")
    print(f"File: {f.name}")
    data = torch.load(f, map_location="cpu", weights_only=False)
    print(f"Keys: {list(data.keys())}")
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, min={v.min():.3f}, max={v.max():.3f}")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")
