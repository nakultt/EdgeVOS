"""
EdgeVOS — Month 1: Distillation Training Script
Distill SAM 3.1 teacher → RepViT-M1 backbone + MLP mask head

Run:
  python train/distill_mask.py --config-name base
"""

import os, torch, wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Internal
from models.mask_head import MLPMaskHead, distillation_loss
from data.davis_dataset import COCODistillDataset


class StudentModel(torch.nn.Module):
    """RepViT-M1 backbone + MLP mask head."""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            "repvit_m1", pretrained=True, features_only=True
        )
        self.mask_head = MLPMaskHead(
            in_dim=384,
            hidden_dims=tuple(cfg.model.mask_head_dims[1:]),
            out_size=cfg.model.mask_out_size,
        )
        # Project backbone output → 384 dim for mask head input
        backbone_out_dim = self.backbone.feature_info[-1]["num_chs"]
        self.adapter = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(backbone_out_dim, 384),
            torch.nn.LayerNorm(384),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, 224, 224] → [B, 1, 256, 256] mask logits"""
        feats   = self.backbone(x)[-1]     # Last feature map
        roi_feat = self.adapter(feats)      # [B, 384]
        return self.mask_head(roi_feat)     # [B, 1, 256, 256]


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # ── Setup ──────────────────────────────────────────────────────
    torch.manual_seed(cfg.project.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project = cfg.wandb.project,
        entity  = cfg.wandb.entity,
        config  = OmegaConf.to_container(cfg, resolve=True),
        name    = "distill_mask_repvit_m1",
    )

    # ── Data ───────────────────────────────────────────────────────
    dataset = COCODistillDataset(
        coco_root   = cfg.data.coco_root,
        targets_dir = cfg.data.sam3_targets_dir,
        img_size    = cfg.data.img_size,
    )
    loader = DataLoader(
        dataset,
        batch_size  = cfg.distill.batch_size,
        shuffle     = True,
        num_workers = cfg.data.num_workers,
        pin_memory  = True,
        drop_last   = True,
    )

    # ── Model + Optimiser ──────────────────────────────────────────
    model = StudentModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.distill.lr,
        weight_decay = cfg.distill.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.distill.epochs, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler()

    # ── Training loop ──────────────────────────────────────────────
    best_loss = float("inf")
    for epoch in range(cfg.distill.epochs):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}")):
            images       = batch["image"].to(device)
            soft_targets = batch["soft_logits"].to(device).float()

            # Create pseudo GT mask from soft targets
            gt_masks = (soft_targets.sigmoid() > 0.5).float()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred_logits = model(images)
                # Resize soft targets to match pred size if needed
                if soft_targets.shape != pred_logits.shape:
                    soft_targets = F.interpolate(
                        soft_targets, size=pred_logits.shape[-2:],
                        mode="bilinear", align_corners=False
                    )
                    gt_masks = F.interpolate(
                        gt_masks, size=pred_logits.shape[-2:], mode="nearest"
                    )
                loss = distillation_loss(
                    pred_logits, soft_targets, gt_masks,
                    soft_weight    = cfg.distill.loss_soft_weight,
                    dice_weight    = cfg.distill.loss_dice_weight,
                    boundary_weight= cfg.distill.loss_boundary_weight,
                ).mean()

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            if step % cfg.wandb.log_interval == 0:
                wandb.log({
                    "loss/step":    loss.item(),
                    "epoch":        epoch,
                    "lr":           scheduler.get_last_lr()[0],
                })

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} | avg_loss={avg_loss:.4f}")
        wandb.log({"loss/epoch": avg_loss, "epoch": epoch})

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "loss":       avg_loss,
                "cfg":        OmegaConf.to_container(cfg),
            }, "checkpoints/best_distill.pt")
            print(f"  Saved best checkpoint (loss={avg_loss:.4f})")

    wandb.finish()
    print("Distillation training complete.")


if __name__ == "__main__":
    main()
