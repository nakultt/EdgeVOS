"""
EdgeVOS — Module 1: Text Grounding (text prompt → candidate boxes + masks)

Two interchangeable grounders:

  SAM3Grounder  — SAM 3 open-vocabulary detector/segmenter (the M1 teacher).
                  Offline / development reference only — not part of the product.
  CLIPGrounder  — MobileCLIP2-S2 dense heatmap in one forward pass.
                  SAM-free product path; boxes refined by the distilled mask head.

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

    provides_masks = True

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
    SAM-free text grounding with MobileCLIP2-S2 in one forward pass.

    FastViT ends in global-average-pool → linear, so applying that linear
    head at every location of the final feature map gives per-location CLIP
    embeddings whose mean is exactly the image embedding. Scoring them against
    the prompt (minus the best background prompt) yields a dense heatmap; each
    peak region becomes a candidate box. Boxes are coarse (32-px grid) and are
    tightened downstream by the distilled mask head.
    """

    provides_masks = False
    BACKGROUND_PROMPTS = ("background", "wall", "floor", "ground", "grass", "sky",
                          "furniture", "stairs", "a plant", "a building")

    def __init__(self, clip_model, preprocess, tokenizer, device: str = "cuda",
                 long_side: int = 1024, presence_threshold: float = 0.14, max_candidates: int = 3):
        self.clip_model = clip_model
        self.trunk = clip_model.visual.trunk
        self.tokenizer = tokenizer
        self.device = device
        self.long_side = long_side
        self.presence_threshold = presence_threshold
        self.max_candidates = max_candidates
        self.mean = torch.zeros(1, 3, 1, 1, device=device)
        self.std = torch.ones(1, 3, 1, 1, device=device)
        for t in getattr(preprocess, "transforms", []):
            if hasattr(t, "mean") and hasattr(t, "std"):
                self.mean = torch.tensor(t.mean, device=device).view(1, 3, 1, 1)
                self.std = torch.tensor(t.std, device=device).view(1, 3, 1, 1)
        self._text_cache = {}
        self._bg_feat = self._encode([f"a photo of {p}." for p in self.BACKGROUND_PROMPTS])

    @torch.no_grad()
    def _encode(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        return F.normalize(self.clip_model.encode_text(tokens).float(), dim=-1)

    def _text_feat(self, prompt: str) -> torch.Tensor:
        if prompt not in self._text_cache:
            self._text_cache[prompt] = self._encode([f"a photo of a {prompt}."])
        return self._text_cache[prompt]

    @torch.no_grad()
    def heatmap(self, frame: torch.Tensor, prompt: str) -> torch.Tensor:
        """Prompt-vs-background similarity margin on the feature grid → [h, w]."""
        H, W = frame.shape[1:]
        s = self.long_side / max(H, W)
        th = max(32, int(round(H * s / 32)) * 32)
        tw = max(32, int(round(W * s / 32)) * 32)
        x = F.interpolate(frame[None].float().to(self.device), size=(th, tw), mode="bilinear", align_corners=False)
        fmap = self.trunk.forward_features((x - self.mean) / self.std)            # [1, C, h, w]
        dense = F.normalize(self.trunk.head.fc(fmap.permute(0, 2, 3, 1)).float(), dim=-1)[0]
        sim = (dense @ self._text_feat(prompt).T)[..., 0]
        bg = (dense @ self._bg_feat.T).max(-1).values
        return sim - bg

    @torch.no_grad()
    def detect(self, frame: torch.Tensor, prompt: str) -> List[Detection]:
        import cv2

        H, W = frame.shape[1:]
        hm = self.heatmap(frame, prompt)
        if float(hm.max()) < self.presence_threshold:
            return []
        up = F.interpolate(hm[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        lo, hi = float(up.min()), float(up.max())
        region = (up >= lo + 0.6 * (hi - lo)).cpu().numpy().astype(np.uint8)
        n, labels = cv2.connectedComponents(region, connectivity=8)
        labels_t = torch.from_numpy(labels).to(self.device)

        dets = []
        for k in range(1, n):
            comp = labels_t == k
            peak = float(up[comp].max())
            if peak < self.presence_threshold:
                continue
            ys, xs = torch.where(comp)
            box = torch.tensor([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                               dtype=torch.float32, device=self.device)
            dets.append(Detection(box=box, score=peak, mask=None))
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets[:self.max_candidates]
