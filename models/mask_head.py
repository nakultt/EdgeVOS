"""
EdgeVOS — Module 2: MLP Mask Head (distilled from SAM 3.1 mask decoder)
3-layer MLP: 384 → 512 → 256 → 256x256
1.8MB vs SAM 3.1 mask decoder's 12MB+ transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        threshold:      float = 0.5,
    ) -> torch.Tensor:
        """
        Full inference: feature → upsampled binary mask at original resolution.
        Args:
            roi_feat:      [B, 384]
            original_size: (H, W) of original frame
            threshold:     binarization threshold
        Returns:
            mask: [B, 1, H, W] bool tensor
        """
        logits = self.forward(roi_feat)                           # [B,1,256,256]
        upsampled = F.interpolate(
            logits, size=original_size, mode='bilinear', align_corners=False
        )                                                          # [B,1,H,W]
        return upsampled.sigmoid() > threshold


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
    loss_soft = F.mse_loss(pred_logits, teacher_logits.detach())
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
    return F.binary_cross_entropy(boundary_pred, boundary_gt)
