"""
EdgeVOS — Module 2: MLP Mask Head (distilled from SAM 3.1 mask decoder)
3-layer MLP: 384 → 512 → 256 → 256x256
1.8MB vs SAM 3.1 mask decoder's 12MB+ transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MLPMaskHead(nn.Module):
    """
    Lightweight mask head distilled from SAM 3.1's mask decoder.
    Input: 384-dim RoI feature (DINOv3 patches mean-pooled inside bbox)
    Output: 256x256 soft mask logits (sigmoid → binary mask)
    """

    def __init__(
        self,
        in_dim:    int = 384,
        hidden_dims = (512, 256),
        out_size:  int = 256,
    ):
        super().__init__()
        self.out_size = out_size
        dims = [in_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.GELU()]
        layers.append(nn.Linear(dims[-1], out_size * out_size))
        self.net = nn.Sequential(*layers)

    def forward(self, roi_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            roi_feat: [B, 384] feature for each RoI (mean-pooled DINOv3 patches)
        Returns:
            logits: [B, 1, out_size, out_size] — apply sigmoid for probability
        """
        B = roi_feat.shape[0]
        return self.net(roi_feat).reshape(B, 1, self.out_size, self.out_size)

    def predict_mask(
        self,
        roi_feat:       torch.Tensor,
        original_size:  tuple,
        bbox:           Optional[torch.Tensor] = None,
        threshold:      float = 0.5,
    ) -> torch.Tensor:
        """
        Full inference: feature → upsampled binary mask at original resolution.
        If bbox is provided, the mask is resized and pasted into the bbox region.
        Args:
            roi_feat:      [B, 384]
            original_size: (H, W) of original frame
            bbox:          [B, 4] bounding boxes in absolute coordinates [x1, y1, x2, y2]
            threshold:     binarization threshold
        Returns:
            mask: [B, 1, H, W] bool tensor
        """
        logits = self.forward(roi_feat)                           # [B,1,256,256]
        B = logits.shape[0]
        H, W = original_size
        
        if bbox is not None:
            # Output mask of shape [B, 1, H, W]
            full_mask = torch.zeros((B, 1, H, W), dtype=torch.bool, device=logits.device)
            # Apply sigmoid to logits to get probabilities
            probs = logits.sigmoid()
            
            for i in range(B):
                x1, y1, x2, y2 = bbox[i].tolist()
                x1 = max(0, min(W - 1, int(round(x1))))
                y1 = max(0, min(H - 1, int(round(y1))))
                x2 = max(x1 + 1, min(W, int(round(x2))))
                y2 = max(y1 + 1, min(H, int(round(y2))))
                
                box_w = x2 - x1
                box_h = y2 - y1
                
                # Resize the 256x256 probability map to the bbox size
                box_prob = F.interpolate(
                    probs[i:i+1], size=(box_h, box_w), mode='bilinear', align_corners=False
                )
                box_mask = box_prob.squeeze(0).squeeze(0) > threshold
                full_mask[i, 0, y1:y2, x1:x2] = box_mask
            return full_mask
        else:
            upsampled = F.interpolate(
                logits, size=original_size, mode='bilinear', align_corners=False
            )                                                          # [B,1,H,W]
            return upsampled.sigmoid() > threshold


class SpatialMaskHead(nn.Module):
    """
    Progressive convolutional decoder for spatial feature maps.
    
    Architecture: 14×14 → 28×28 → 56×56 → 112×112 → 256×256
    Each stage: upsample 2× → Conv3×3 + BN + GELU
    
    This interleaving of upsampling and convolutions preserves spatial
    detail at every resolution, producing object-shaped masks instead
    of rectangles.
    
    Input:  [B, C, 14, 14]   (C = 384 for RepViT-M1)
    Output: [B, 1, out_size, out_size] mask logits
    """

    def __init__(self, in_channels: int = 384, out_size: int = 256):
        super().__init__()
        self.out_size = out_size

        # Project from backbone channels to decoder width
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # Stage 1: 14×14 → 28×28  (256 → 128 ch)
        self.up1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        # Stage 2: 28×28 → 56×56  (128 → 64 ch)
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        # Stage 3: 56×56 → 112×112  (64 → 32 ch)
        self.up3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        # Final head: 1×1 conv to single-channel mask logits
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_map: [B, C, 14, 14] spatial features from backbone
        Returns:
            [B, 1, out_size, out_size] mask logits (apply sigmoid for probs)
        """
        x = self.proj(feat_map)                                       # [B, 256, 14, 14]

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)                                               # [B, 128, 28, 28]

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)                                               # [B, 64, 56, 56]

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x)                                               # [B, 32, 112, 112]

        x = self.head(x)                                              # [B, 1, 112, 112]

        # Final resize to target output size (112 → 256)
        if x.shape[-2:] != (self.out_size, self.out_size):
            x = F.interpolate(
                x, size=(self.out_size, self.out_size),
                mode="bilinear", align_corners=False
            )
        return x


def distillation_loss(
    pred_logits:    torch.Tensor,   # [B, 1, 256, 256] student output
    teacher_logits: torch.Tensor,   # [B, 1, 256, 256] SAM 3.1 soft targets
    gt_masks:       torch.Tensor,   # [B, 1, 256, 256] hard binary GT
    soft_weight:    float = 1.0,
    dice_weight:    float = 0.5,
    boundary_weight: float = 0.2,
) -> torch.Tensor:
    """
    Combined distillation loss:
      1. Soft: MSE on SAM 3.1 logits (match teacher distribution)
      2. Dice: quality of predicted binary mask
      3. Boundary: preserve sharp edges (SAM 3.1 improved boundary quality)
    """
    # Foreground-weighted MSE — background pixels get 0.1 weight
    # Otherwise 88% background dominates and foreground never learns
    fg_mask = (gt_masks > 0.5).float()
    bg_mask = 1.0 - fg_mask
    pixel_weights = fg_mask * 1.0 + bg_mask * 0.1   # foreground 10x more important

    # Apply pixel weights to the MSE loss
    # using .mean() directly after multiplication averages over all pixels
    loss_soft = (pixel_weights * (pred_logits - teacher_logits.detach())**2).mean()
    
    loss_dice = _dice_loss(pred_logits.sigmoid(), gt_masks.float())
    loss_bnd  = _boundary_loss(pred_logits.sigmoid(), gt_masks.float())
    
    return soft_weight * loss_soft + dice_weight * loss_dice + boundary_weight * loss_bnd


def _dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred   = pred.flatten(1)
    target = target.flatten(1)
    inter  = (pred * target).sum(1)
    denom  = pred.sum(1) + target.sum(1)
    return 1 - (2 * inter + eps) / (denom + eps)


def _boundary_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize errors near object boundaries."""
    laplacian = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=pred.dtype, device=pred.device
    ).view(1, 1, 3, 3)
    boundary_gt   = F.conv2d(target,   laplacian, padding=1).abs().clamp(0, 1)
    boundary_pred = F.conv2d(pred,     laplacian, padding=1).abs().clamp(0, 1)
    with torch.amp.autocast('cuda', enabled=False):
        return F.binary_cross_entropy(boundary_pred.float(), boundary_gt.float())
