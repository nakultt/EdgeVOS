"""
EdgeVOS — Module 1: Student Backbone with RoI Feature Extraction

Architecture:
  - RepViT-M1 (via timm) as the efficient student backbone
  - Adaptive average pool → 14×14 spatial grid (matching DINOv3's patch layout)
  - torchvision RoI Align to extract region features from that grid
  - Outputs 384-dim RoI vectors consumed by MLPMaskHead (Module 2)
  - Also exposes flat [B, 196, 384] patch features for STGP (Module 3)
 
RepViT paper:  https://arxiv.org/abs/2307.09283
timm docs:     https://huggingface.co/docs/timm/models/repvit
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from typing import List, Optional, Tuple

import timm


class BackboneWithRoI(nn.Module):
    """
    Student backbone for EdgeVOS.

    Wraps timm's ``repvit_m1`` to produce a 14×14×384 feature map,
    then uses RoI Align to pool object-level features from bounding boxes.

    Outputs
    -------
    forward(images, bboxes) → [B, 384]
        Mean-pooled RoI features ready for :class:`MLPMaskHead`.
    get_patch_features(images) → [B, 196, 384]
        Flattened patch tokens for :class:`STGPGraph` propagation.
    """

    # Spatial size of the feature grid (matches DINOv3 14×14 patch layout)
    GRID_SIZE: int = 14
    # Channel dimension of RepViT-M1's final feature map
    FEAT_DIM: int = 384

    def __init__(
        self,
        model_name: str = "repvit_m1",
        pretrained: bool = True,
        grid_size: int = 14,
        roi_output_size: int = 7,
    ) -> None:
        """
        Parameters
        ----------
        model_name : str
            timm model identifier.  Default ``repvit_m1``.
        pretrained : bool
            Load ImageNet-1K pretrained weights.
        grid_size : int
            Spatial size to which the backbone feature map is adaptively
            pooled.  Must be 14 to match DINOv3 patch count (196 = 14²).
        roi_output_size : int
            Output spatial size of :func:`roi_align`.  The aligned region
            is global-average-pooled to a single 384-dim vector afterwards.
        """
        super().__init__()
        self.grid_size = grid_size
        self.roi_output_size = roi_output_size

        # ── Backbone ────────────────────────────────────────────────
        # features_only=True returns intermediate feature maps; we take
        # the last one which has the richest semantic content.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
        )

        # Adaptive pool to a fixed 14×14 grid regardless of input resolution.
        self.adapt_pool = nn.AdaptiveAvgPool2d(grid_size)

        # Resolve actual channel dim from the backbone's feature_info
        # (RepViT-M1 last stage = 384 channels — verify at runtime).
        self._feat_channels: Optional[int] = None

    # ── Properties ──────────────────────────────────────────────────

    @property
    def feat_channels(self) -> int:
        """Number of channels in the last feature map (resolved lazily)."""
        if self._feat_channels is not None:
            return self._feat_channels
        info = self.backbone.feature_info.channels()
        self._feat_channels = info[-1]
        return self._feat_channels

    # ── Feature extraction ──────────────────────────────────────────

    def _extract_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run the backbone and return the adaptively-pooled feature map.

        Parameters
        ----------
        images : Tensor [B, 3, H, W]
            Batch of RGB images (ImageNet-normalised).

        Returns
        -------
        Tensor [B, C, grid_size, grid_size]
            Feature map, where C = :pyattr:`feat_channels` (384 for RepViT-M1).
        """
        feats: List[torch.Tensor] = self.backbone(images)
        last_feat = feats[-1]  # highest-level feature map
        return self.adapt_pool(last_feat)  # [B, C, 14, 14]

    # ── Public API ──────────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract 384-dim RoI features for each bounding box.

        Parameters
        ----------
        images : Tensor [B, 3, H, W]
            Batch of input images.
        bboxes : Tensor [B, 4]
            Bounding boxes in **normalised** ``[x1, y1, x2, y2]`` format,
            values in ``[0, 1]`` relative to image size.  One box per image
            (matching the 1-object-per-crop distillation setup).

        Returns
        -------
        Tensor [B, 384]
            RoI features ready for :class:`MLPMaskHead`.
        """
        feat_map = self._extract_feature_map(images)  # [B, C, 14, 14]
        B = feat_map.shape[0]
        G = self.grid_size  # 14

        # Scale normalised bboxes to the feature grid coordinate space.
        # roi_align expects boxes in (x1, y1, x2, y2) absolute coords on
        # the *feature map*, plus a batch index column.
        boxes_scaled = bboxes.clone().float()
        boxes_scaled[:, [0, 2]] *= G  # x coords
        boxes_scaled[:, [1, 3]] *= G  # y coords

        # Build [B, 5] tensor: (batch_idx, x1, y1, x2, y2)
        batch_idx = torch.arange(B, device=bboxes.device, dtype=torch.float32).unsqueeze(1)
        rois = torch.cat([batch_idx, boxes_scaled], dim=1)  # [B, 5]

        # RoI Align on the 14×14 feature map
        aligned: torch.Tensor = roi_align(
            feat_map,
            rois,
            output_size=self.roi_output_size,
            spatial_scale=1.0,  # boxes already in feature-map coords
            aligned=True,
        )  # [B, C, roi_output_size, roi_output_size]

        # Global average pool → [B, C]
        roi_feat = aligned.mean(dim=[2, 3])  # [B, 384]

        return roi_feat

    def get_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Return flattened patch-level features for STGP graph propagation.

        Parameters
        ----------
        images : Tensor [B, 3, H, W]
            Batch of input images.

        Returns
        -------
        Tensor [B, 196, 384]
            Patch tokens reshaped from the 14×14 feature grid.
        """
        feat_map = self._extract_feature_map(images)  # [B, C, 14, 14]
        B, C, H, W = feat_map.shape
        # [B, C, 14, 14] → [B, 196, C]
        return feat_map.flatten(2).transpose(1, 2)  # [B, 196, 384]


# ════════════════════════════════════════════════════════════════════
# Unit tests
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[backbone] Running unit tests on {device}")

    # ── Build model (pretrained=False for CI speed) ─────────────
    model = BackboneWithRoI(pretrained=False).to(device)
    print(f"[backbone] Feature channels = {model.feat_channels}")

    B = 4
    images = torch.randn(B, 3, 224, 224, device=device)

    # ── Test get_patch_features ─────────────────────────────────
    patches = model.get_patch_features(images)
    assert patches.shape == (B, 196, 384), (
        f"Expected (4, 196, 384), got {patches.shape}"
    )
    print(f"[backbone] get_patch_features -> {patches.shape}  OK")

    # ── Test forward (RoI) ──────────────────────────────────────
    # Normalised bboxes: small region in the centre of each image
    bboxes = torch.tensor(
        [[0.2, 0.2, 0.8, 0.8]] * B,
        device=device,
        dtype=torch.float32,
    )
    roi_feats = model(images, bboxes)
    assert roi_feats.shape == (B, 384), (
        f"Expected (4, 384), got {roi_feats.shape}"
    )
    print(f"[backbone] forward (RoI)       -> {roi_feats.shape}  OK")

    # ── Edge case: full-image bbox ──────────────────────────────
    bboxes_full = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0]] * B,
        device=device,
        dtype=torch.float32,
    )
    roi_full = model(images, bboxes_full)
    assert roi_full.shape == (B, 384)
    print(f"[backbone] full-image RoI      -> {roi_full.shape}  OK")

    # ── Edge case: tiny bbox ────────────────────────────────────
    bboxes_tiny = torch.tensor(
        [[0.49, 0.49, 0.51, 0.51]] * B,
        device=device,
        dtype=torch.float32,
    )
    roi_tiny = model(images, bboxes_tiny)
    assert roi_tiny.shape == (B, 384)
    print(f"[backbone] tiny RoI            -> {roi_tiny.shape}  OK")

    # ── Verify grad flow ────────────────────────────────────────
    loss = roi_feats.sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters() if p.requires_grad)
    assert has_grad, "No gradients flowed through the model!"
    print("[backbone] Gradient flow        OK")

    print("\n[backbone] All tests passed!")
    sys.exit(0)
