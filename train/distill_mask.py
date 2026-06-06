"""
EdgeVOS — Month 1: Distillation Training Script
Distill SAM 3.1 teacher → RepViT-M1 backbone + MLP mask head

Run:
  python train/distill_mask.py --config-name base
"""

import os
import sys
import torch
import wandb
import torch.nn.functional as F
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader, Subset
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Internal
from models.backbone import BackboneWithRoI
from models.mask_head import MLPMaskHead, distillation_loss
from data.davis_dataset import COCODistillDataset


class StudentModel(torch.nn.Module):
    """RepViT-M1 image-to-mask student."""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = BackboneWithRoI(
            model_name=cfg.model.backbone,
            pretrained=cfg.model.backbone_pretrained,
        )
        self.mask_head = MLPMaskHead(
            in_dim=self.backbone.feat_channels,
            hidden_dims=cfg.model.mask_head_dims[1:],  # (512, 256)
            out_size=cfg.model.mask_out_size,          # 256
        )

    def forward(self, images: torch.Tensor, bboxes: torch.Tensor = None) -> torch.Tensor:
        """
        images: [B, 3, 224, 224]
        bboxes: [B, 4] normalized coords [x1, y1, x2, y2]. Default: centered crop.
        Returns: [B, 1, 256, 256] mask logits
        """
        B = images.shape[0]
        if bboxes is None:
            # Full crop region for distillation (images are already cropped to object)
            bboxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * B, device=images.device)
            
        roi_feat = self.backbone(images, bboxes)  # [B, 384]
        return self.mask_head(roi_feat)           # [B, 1, 256, 256]


def compute_iou(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    """Compute IoU between two boolean masks [B, 1, H, W]"""
    intersection = (mask1 & mask2).sum((1, 2, 3)).float()
    union = (mask1 | mask2).sum((1, 2, 3)).float()
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # ── Setup ──────────────────────────────────────────────────────
    torch.manual_seed(cfg.project.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.wandb.entity is not None or "edgevos" in cfg.wandb.project:
        wandb.init(
            project = cfg.wandb.project,
            entity  = cfg.wandb.entity,
            config  = OmegaConf.to_container(cfg, resolve=True),
            name    = "distill_mask_repvit_m1",
        )
    else:
        wandb.init(
            project = cfg.wandb.project,
            config  = OmegaConf.to_container(cfg, resolve=True),
            name    = "distill_mask_repvit_m1",
            mode    = "offline"
        )

    # ── Data ───────────────────────────────────────────────────────
    full_dataset = COCODistillDataset(
        coco_root   = cfg.data.coco_root,
        targets_dir = cfg.data.sam3_targets_dir,
        split       = "train2017",
        img_size    = cfg.data.img_size,
    )
    
    # Validation split (100 images)
    val_indices = list(range(min(100, len(full_dataset))))
    val_dataset = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(
        full_dataset,
        batch_size  = cfg.distill.batch_size,
        shuffle     = True,
        num_workers = cfg.data.num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg.distill.batch_size,
        shuffle     = False,
        num_workers = cfg.data.num_workers,
        pin_memory  = True,
        drop_last   = False,
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
    scaler = torch.amp.GradScaler('cuda')

    # ── Training loop ──────────────────────────────────────────────
    best_iou = 0.0
    epochs_without_improvement = 0
    patience = 5
    
    summary_data = []

    for epoch in range(cfg.distill.epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.distill.epochs}")
        for step, batch in enumerate(pbar):
            images       = batch["image"].to(device)
            soft_targets = batch["soft_logits"].to(device).float()

            # Create pseudo GT mask from soft targets
            gt_masks = (soft_targets.sigmoid() > 0.5).float()

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred_logits = model(images)
                if soft_targets.shape != pred_logits.shape:
                    soft_targets = F.interpolate(
                        soft_targets, size=pred_logits.shape[-2:],
                        mode="bilinear", align_corners=False
                    )
                    gt_masks = F.interpolate(
                        gt_masks, size=pred_logits.shape[-2:], mode="nearest"
                    )
                
                # Foreground-weighted BCE (Teacher soft targets are probabilities, so apply sigmoid)
                fg_mask = (gt_masks > 0.5).float()
                bg_mask = 1.0 - fg_mask
                pixel_weights = fg_mask * 1.0 + bg_mask * 0.1
                
                loss_soft = F.binary_cross_entropy_with_logits(
                    pred_logits, soft_targets.sigmoid().detach(), weight=pixel_weights
                )
                
                loss_dice = _dice_loss(pred_logits.sigmoid(), gt_masks)
                loss_bnd  = _boundary_loss(pred_logits.sigmoid(), gt_masks)
                
                loss = (cfg.distill.loss_soft_weight * loss_soft + \
                       cfg.distill.loss_dice_weight * loss_dice + \
                       cfg.distill.loss_boundary_weight * loss_bnd).mean()

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Warmup LR adjustment
            if epoch < cfg.distill.warmup_epochs:
                warmup_factor = (epoch * len(train_loader) + step) / (cfg.distill.warmup_epochs * len(train_loader))
                lr = 1e-6 + (cfg.distill.lr - 1e-6) * warmup_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # W&B Logging
            global_step = epoch * len(train_loader) + step
            if global_step % cfg.wandb.log_interval == 0:
                wandb.log({
                    "train/loss_step": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                })
                
            if global_step % cfg.wandb.log_images_every == 0:
                # Log 4 sample images
                n_samples = min(4, images.shape[0])
                log_images = []
                for i in range(n_samples):
                    img_display = images[i].permute(1, 2, 0).cpu().numpy()
                    pred_mask = (pred_logits[i].sigmoid() > 0.5).squeeze().cpu().numpy()
                    teacher_mask = gt_masks[i].squeeze().cpu().numpy()
                    
                    # Log to W&B
                    log_images.append(wandb.Image(
                        img_display, 
                        masks={
                            "prediction": {"mask_data": pred_mask, "class_labels": {1: "Student"}},
                            "ground_truth": {"mask_data": teacher_mask, "class_labels": {1: "Teacher"}}
                        },
                        caption=f"Sample {i}"
                    ))
                wandb.log({"train/predictions": log_images, "epoch": epoch})

        if epoch >= cfg.distill.warmup_epochs:
            scheduler.step()
            
        avg_loss = total_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_iou_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                soft_targets = batch["soft_logits"].to(device).float()
                teacher_masks = soft_targets.sigmoid() > 0.5
                
                pred_logits = model(images)
                student_masks = pred_logits.sigmoid() > 0.5
                
                val_iou_sum += compute_iou(student_masks, teacher_masks) * images.size(0)
                
        val_iou = val_iou_sum / len(val_dataset)
        vs_teacher = val_iou # Since the target IS the teacher mask, IoU == vs_teacher ratio
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val IoU: {val_iou:.4f} | vs Teacher: {vs_teacher:.4f}")
        wandb.log({"train/loss_epoch": avg_loss, "val/iou": val_iou, "val/vs_teacher": vs_teacher, "epoch": epoch})
        
        summary_data.append([epoch+1, f"{avg_loss:.4f}", f"{val_iou:.4f}", f"{vs_teacher:.4f}"])

        # Early stopping and Checkpointing
        if val_iou > best_iou:
            best_iou = val_iou
            epochs_without_improvement = 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_iou":    val_iou,
                "cfg":        OmegaConf.to_container(cfg),
            }, "checkpoints/best_distill.pt")
            print(f"  Saved best checkpoint (Val IoU={val_iou:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered. Validation IoU hasn't improved for {patience} epochs.")
                break

    wandb.finish()
    
    # Print Summary Table
    print("\n" + "="*50)
    print("--- Training Summary ---")
    print(f"{'Epoch':<10} | {'Train Loss':<15} | {'Val IoU':<15} | {'vs Teacher':<15}")
    print("-" * 65)
    for row in summary_data:
        print(f"{row[0]:<10} | {row[1]:<15} | {row[2]:<15} | {row[3]:<15}")
    print("="*50 + "\n")
    
    print("\nDistillation training complete.")


if __name__ == "__main__":
    main()
