"""
EdgeVOS — Module 1: Text Grounding (text prompt → candidate boxes + masks)

Two interchangeable grounders:

  SAM3Grounder  — SAM 3 open-vocabulary detector/segmenter (the M1 teacher).
                  Accurate; used at initialisation and for re-grounding.
  CLIPGrounder  — MobileCLIP2-S2 dense multi-scale window scoring.
                  Lightweight fallback when SAM 3 is not available.

Both return a list of ``Detection`` sorted by score (best first).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SAM3_REPO = os.path.join(_PROJECT_ROOT, "sam3")


@dataclass
class Detection:
    box:   torch.Tensor            # [4] absolute xyxy (float, on device)
    score: float                   # grounding confidence
    mask:  Optional[torch.Tensor]  # [H, W] bool, or None if the grounder has no masks


class SAM3Grounder:
    """Wraps ``sam3.Sam3Processor`` for single-image text grounding."""

    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.3):
        if os.path.isdir(_SAM3_REPO) and _SAM3_REPO not in sys.path:
            sys.path.insert(0, _SAM3_REPO)
        import sam3
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        bpe_path = os.path.join(os.path.dirname(sam3.__file__), "assets", "bpe_simple_vocab_16e6.txt.gz")
        if not os.path.exists(bpe_path):
            bpe_path = os.path.join(_SAM3_REPO, "assets", "bpe_simple_vocab_16e6.txt.gz")

        self.device = device
        model = build_sam3_image_model(bpe_path=bpe_path, device=device)
        self.processor = Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)

    @torch.no_grad()
    def detect(self, frame: torch.Tensor, prompt: str) -> List[Detection]:
        """frame: [3, H, W] RGB float in [0, 1]."""
        from PIL import Image

        rgb = (frame.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            state = self.processor.set_image(Image.fromarray(rgb))
            out = self.processor.set_text_prompt(prompt=prompt, state=state)

        dets = [
            Detection(
                box=out["boxes"][i].float().to(self.device),
                score=float(out["scores"][i]),
                mask=out["masks"][i, 0].to(self.device),
            )
            for i in range(len(out["scores"]))
        ]
        return sorted(dets, key=lambda d: d.score, reverse=True)


class CLIPGrounder:
    """
    Dense multi-scale sliding-window localisation with MobileCLIP2-S2.

    Windows are scored against the text embedding in GPU batches and turned
    into a coverage-normalised heatmap; the box is the high-response region.
    Coarser than SAM 3 (no mask, weak on small objects) but far tighter than
    picking a single grid tile.
    """

    def __init__(self, clip_model, preprocess, tokenizer, device: str = "cuda",
                 scales=(0.15, 0.25, 0.35, 0.5, 0.7), stride_ratio: float = 0.33):
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.device = device
        self.scales = scales
        self.stride_ratio = stride_ratio
        # Pull resize/normalisation constants out of the open_clip transform so
        # crops can be batched on the GPU instead of going through PIL.
        self.input_size = 256
        self.mean = torch.tensor([0.0, 0.0, 0.0], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([1.0, 1.0, 1.0], device=device).view(1, 3, 1, 1)
        for t in getattr(preprocess, "transforms", []):
            if hasattr(t, "size") and not hasattr(t, "mean"):
                s = t.size
                self.input_size = s if isinstance(s, int) else s[0]
            if hasattr(t, "mean") and hasattr(t, "std"):
                self.mean = torch.tensor(t.mean, device=device).view(1, 3, 1, 1)
                self.std = torch.tensor(t.std, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def _text_feat(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer([f"a photo of a {prompt}"]).to(self.device)
        with torch.autocast("cuda", enabled=self.device.startswith("cuda")):
            return F.normalize(self.clip_model.encode_text(tokens).float(), dim=-1)

    def _windows(self, H: int, W: int) -> List[List[int]]:
        boxes = []
        side = min(H, W)
        for s in self.scales:
            for aspect in (1.0, 1.5, 0.67):
                bw = int(min(W, side * s * aspect ** 0.5))
                bh = int(min(H, side * s / aspect ** 0.5))
                sx = max(1, int(bw * self.stride_ratio))
                sy = max(1, int(bh * self.stride_ratio))
                for y in range(0, H - bh + 1, sy):
                    for x in range(0, W - bw + 1, sx):
                        boxes.append([x, y, x + bw, y + bh])
        return boxes

    @torch.no_grad()
    def detect(self, frame: torch.Tensor, prompt: str) -> List[Detection]:
        H, W = frame.shape[1:]
        text_feat = self._text_feat(prompt)
        boxes = self._windows(H, W)
        frame = frame.to(self.device).unsqueeze(0)

        scores = []
        for i in range(0, len(boxes), 64):
            crops = torch.cat([
                F.interpolate(frame[:, :, y1:y2, x1:x2], size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)
                for x1, y1, x2, y2 in boxes[i:i + 64]
            ])
            crops = (crops - self.mean) / self.std
            with torch.autocast("cuda", enabled=self.device.startswith("cuda")):
                img_feat = F.normalize(self.clip_model.encode_image(crops).float(), dim=-1)
            scores.append((img_feat @ text_feat.T).squeeze(1))
        scores = torch.cat(scores)

        # Coverage-normalised heatmap: each pixel averages the (sharpened) scores
        # of the windows covering it, so context-heavy large windows don't win.
        weights = torch.exp((scores - scores.max()) * 50.0).tolist()
        heat = torch.zeros(H, W, device=self.device)
        cover = torch.full((H, W), 1e-6, device=self.device)
        for (x1, y1, x2, y2), w in zip(boxes, weights):
            heat[y1:y2, x1:x2] += w
            cover[y1:y2, x1:x2] += 1
        heat = heat / cover
        ys, xs = torch.where(heat >= 0.5 * heat.max())
        box = torch.tensor([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                           dtype=torch.float32, device=self.device)
        return [Detection(box=box, score=float(scores.max()), mask=None)]
