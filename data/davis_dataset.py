"""
EdgeVOS — DAVIS-2017 Dataset Loader
Supports: training sequences, validation evaluation, single-sequence iteration
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
import cv2


class DAVISDataset(Dataset):
    """
    DAVIS-2017 dataset for video object segmentation.
    Returns full video sequences for temporal training/evaluation.
    """

    def __init__(
        self,
        root:       str,
        split:      str = "train",          # "train" | "val"
        resolution: str = "480p",
        img_size:   int = 224,              # Resize for model input
        max_frames: int = 100,              # Truncate very long sequences
    ):
        self.root       = Path(root)
        self.resolution = resolution
        self.img_size   = img_size
        self.max_frames = max_frames

        split_file = self.root / "ImageSets" / "2017" / f"{split}.txt"
        self.sequences = sorted(split_file.read_text().strip().split("\n"))
        print(f"DAVIS-2017 {split}: {len(self.sequences)} sequences")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        """Returns one video sequence as a dict of stacked tensors."""
        return self.get_sequence(self.sequences[idx])

    def get_sequence(self, name: str) -> dict:
        img_dir  = self.root / "JPEGImages"  / self.resolution / name
        ann_dir  = self.root / "Annotations" / self.resolution / name

        img_paths  = sorted(img_dir.glob("*.jpg"))[:self.max_frames]
        ann_paths  = sorted(ann_dir.glob("*.png"))[:self.max_frames]

        frames, masks = [], []
        for ip, ap in zip(img_paths, ann_paths):
            # Load and resize frame
            img = Image.open(ip).convert("RGB")
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            frames.append(torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0)

            # Load annotation — palette PNG, object IDs as pixel values
            ann = Image.open(ap)
            ann = ann.resize((self.img_size, self.img_size), Image.NEAREST)
            ann_arr = np.array(ann)
            # Binary mask: any non-background pixel
            mask = torch.from_numpy((ann_arr > 0).astype(np.float32))
            masks.append(mask)

        return {
            "name":   name,
            "frames": torch.stack(frames),   # [T, 3, H, W]
            "masks":  torch.stack(masks),    # [T, H, W]
            "n_frames": len(frames),
        }

    def get_first_frame(self, name: str) -> tuple:
        """Return (frame_tensor, mask_tensor) for frame 0 only. Used at init."""
        seq = self.get_sequence(name)
        return seq["frames"][0], seq["masks"][0]


class COCODistillDataset(Dataset):
    """
    COCO dataset for M2 mask head distillation.
    Returns image + pre-cached SAM 3.1 soft mask targets.
    """

    def __init__(
        self,
        coco_root:      str,
        targets_dir:    str,
        split:          str = "train2017",
        img_size:       int = 224,
    ):
        from pycocotools.coco import COCO
        self.coco     = COCO(f"{coco_root}/annotations/instances_{split}.json")
        self.img_dir  = Path(coco_root) / split
        # Point to the actual subfolder containing the .pt files
        self.targets_root = Path(targets_dir) / split
        self.img_size = img_size

        if not self.targets_root.exists():
            print(f"[WARN] Targets directory not found: {self.targets_root}")
            self.ann_ids = []
        else:
            # Only keep annotation IDs that have pre-generated targets (ann_{id}.pt)
            all_pt_files = list(self.targets_root.glob("ann_*.pt"))
            self.ann_ids = []
            for f in all_pt_files:
                try:
                    # Extract ID from 'ann_1234.pt'
                    aid = int(f.stem.split("_")[1])
                    self.ann_ids.append(aid)
                except (IndexError, ValueError):
                    continue
        
        print(f"COCO distill: {len(self.ann_ids)} annotations with cached targets in {self.targets_root}")

    def __len__(self) -> int:
        return len(self.ann_ids)

    def __getitem__(self, idx: int) -> dict:
        ann_id = self.ann_ids[idx]
        ann_info = self.coco.loadAnns([ann_id])[0]
        img_id   = ann_info["image_id"]
        img_info = self.coco.loadImgs(img_id)[0]

        img = Image.open(self.img_dir / img_info["file_name"]).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0

        # Load pre-cached SAM 3.1 soft targets (ann_{id}.pt)
        target = torch.load(self.targets_root / f"ann_{ann_id}.pt", map_location="cpu")

        return {
            "ann_id":       ann_id,
            "img_id":       img_id,
            "image":        img_tensor,            # [3, 224, 224]
            "soft_logits":  target["soft_logits"], # [1, 256, 256] FP16
            "boxes":        target["boxes"],        # [4] xyxy
        }
