import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm
from transformers import AutoModel
import random

import sys
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.stgp import STGPProjection


class STGPContrastiveDataset(Dataset):
    """
    Dataset for STGP contrastive training.
    Yields positive pairs (img1, mask1, img2, mask2) for the same object in different frames.
    """
    def __init__(self, davis_root, split="train", resolution="480p", img_size=224, min_area=100):
        self.root = Path(davis_root)
        self.resolution = resolution
        self.img_size = img_size
        self.min_area = min_area
        
        split_file = self.root / "ImageSets" / "2017" / f"{split}.txt"
        sequences = sorted(split_file.read_text().strip().split("\n"))
        
        self.tracklets = []
        
        print(f"Parsing DAVIS-2017 {split} for STGP contrastive training...")
        for seq in sequences:
            ann_dir = self.root / "Annotations" / self.resolution / seq
            ann_paths = sorted(ann_dir.glob("*.png"))
            
            # Find all objects and their valid frames in this sequence
            obj_to_frames = {}
            for frame_idx, ap in enumerate(ann_paths):
                ann_arr = np.array(Image.open(ap))
                obj_ids = np.unique(ann_arr)
                
                for obj_id in obj_ids:
                    if obj_id == 0:
                        continue # Background
                    
                    area = (ann_arr == obj_id).sum()
                    if area > self.min_area:
                        if obj_id not in obj_to_frames:
                            obj_to_frames[obj_id] = []
                        obj_to_frames[obj_id].append(frame_idx)
            
            for obj_id, frames in obj_to_frames.items():
                if len(frames) >= 2:
                    # Valid tracklet: we can sample at least 2 distinct frames
                    self.tracklets.append({
                        "seq": seq,
                        "obj_id": obj_id,
                        "valid_frames": frames,
                        "img_dir": self.root / "JPEGImages" / self.resolution / seq,
                        "ann_dir": ann_dir,
                        "img_paths": sorted((self.root / "JPEGImages" / self.resolution / seq).glob("*.jpg")),
                        "ann_paths": ann_paths
                    })
                    
        print(f"Found {len(self.tracklets)} valid object tracklets in {split}.")

    def __len__(self):
        return len(self.tracklets)

    def __getitem__(self, idx):
        tracklet = self.tracklets[idx]
        
        # Sample two distinct frames
        t1, t2 = random.sample(tracklet["valid_frames"], 2)
        
        img1, mask1 = self._load_frame(tracklet, t1)
        img2, mask2 = self._load_frame(tracklet, t2)
        
        return {
            "img1": img1, "mask1": mask1,
            "img2": img2, "mask2": mask2
        }
        
    def _load_frame(self, tracklet, frame_idx):
        img_path = tracklet["img_paths"][frame_idx]
        ann_path = tracklet["ann_paths"][frame_idx]
        
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        
        ann = Image.open(ann_path)
        ann = ann.resize((self.img_size, self.img_size), Image.NEAREST)
        ann_arr = np.array(ann)
        
        mask = torch.from_numpy((ann_arr == tracklet["obj_id"]).astype(np.float32))
        return img_t, mask


def mask_pool_features(patches, masks):
    """
    Mean-pool patch features using binary masks.
    patches: [B, 196, 384]
    masks: [B, 224, 224]
    """
    B = patches.shape[0]
    # Interpolate masks to patch grid size (14x14)
    grid_masks = F.interpolate(masks.unsqueeze(1), size=(14, 14), mode='nearest').view(B, 196) # [B, 196]
    
    pooled = []
    for i in range(B):
        m = grid_masks[i] > 0.5
        inside = patches[i][m]
        if inside.size(0) == 0:
            pooled.append(patches[i].mean(dim=0))
        else:
            pooled.append(inside.mean(dim=0))
            
    return torch.stack(pooled, dim=0) # [B, 384]


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, negatives_per_positive=15):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = negatives_per_positive
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, q, k):
        # q: [B, D] queries (from frame 1)
        # k: [B, D] keys (from frame 2)
        B = q.size(0)
        
        # Calculate full similarity matrix
        sim = torch.matmul(q, k.T) / self.temperature # [B, B]
        
        # We want to use exactly self.num_negatives per positive.
        # But B might be less than self.num_negatives + 1.
        actual_negatives = min(self.num_negatives, B - 1)
        if actual_negatives <= 0:
            # If batch size is 1, loss is 0
            return torch.tensor(0.0, device=q.device, requires_grad=True), 0.0, 0.0
            
        losses = []
        pos_sims = []
        neg_sims = []
        
        for i in range(B):
            pos_sim = sim[i, i].unsqueeze(0)
            
            # Get all negative indices
            neg_indices = [j for j in range(B) if j != i]
            
            # Sample num_negatives randomly
            if len(neg_indices) > actual_negatives:
                neg_indices = random.sample(neg_indices, actual_negatives)
                
            neg_sim = sim[i, neg_indices]
            
            # Logits for cross entropy [1, 1+N]
            logits = torch.cat([pos_sim, neg_sim]).unsqueeze(0)
            labels = torch.zeros(1, dtype=torch.long, device=q.device)
            
            loss = self.cross_entropy(logits, labels)
            losses.append(loss)
            
            pos_sims.append(pos_sim.item() * self.temperature)
            neg_sims.append(neg_sim.mean().item() * self.temperature)
            
        return sum(losses) / B, sum(pos_sims)/B, sum(neg_sims)/B


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize WandB
    if cfg.wandb.entity is not None or "stgp" in cfg.wandb.project:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name="train_stgp_proj",
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    else:
        # Fallback offline or no entity
        wandb.init(
            project=cfg.wandb.project,
            name="train_stgp_proj",
            config=OmegaConf.to_container(cfg, resolve=True),
            mode="offline" if not cfg.wandb.entity else "online"
        )
        
    # Dataset & Loader
    train_dataset = STGPContrastiveDataset(
        davis_root=cfg.data.davis_root,
        split="train",
        resolution=cfg.eval.davis_resolution,
        img_size=cfg.data.img_size
    )
    val_dataset = STGPContrastiveDataset(
        davis_root=cfg.data.davis_root,
        split="val",
        resolution=cfg.eval.davis_resolution,
        img_size=cfg.data.img_size
    )
    
    # Check batch_size
    train_bs = cfg.stgp_train.batch_size
    
    train_loader = DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True,
        num_workers=cfg.data.num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_bs, shuffle=False,
        num_workers=cfg.data.num_workers, drop_last=True
    )
    
    # Models
    dinov3 = AutoModel.from_pretrained(cfg.model.dinov3_model).to(device)
    dinov3.eval()
    
    stgp_proj = STGPProjection(in_dim=cfg.model.dinov3_feat_dim, out_dim=cfg.model.stgp_proj_dim).to(device)
    
    # Optimizer & Loss
    optimizer = torch.optim.AdamW(stgp_proj.parameters(), lr=cfg.stgp_train.lr)
    criterion = InfoNCELoss(
        temperature=cfg.stgp_train.temperature, 
        negatives_per_positive=cfg.stgp_train.negatives_per_positive
    )
    
    epochs = cfg.stgp_train.epochs
    
    best_val_loss = float('inf')
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    
    print("Starting STGP Contrastive Training...")
    
    for epoch in range(epochs):
        stgp_proj.train()
        train_loss = 0.0
        train_pos_sim = 0.0
        train_neg_sim = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            img1 = batch["img1"].to(device)
            mask1 = batch["mask1"].to(device)
            img2 = batch["img2"].to(device)
            mask2 = batch["mask2"].to(device)
            
            with torch.no_grad():
                # DINOv3 uses 1 CLS + 4 register tokens. The 196 patch tokens are at the end.
                out1 = dinov3(img1).last_hidden_state[:, -196:, :] # [B, 196, 384]
                out2 = dinov3(img2).last_hidden_state[:, -196:, :]
                
                feat1_384 = mask_pool_features(out1, mask1) # [B, 384]
                feat2_384 = mask_pool_features(out2, mask2)
                
            # Forward
            optimizer.zero_grad()
            q = stgp_proj(feat1_384) # [B, 256]
            k = stgp_proj(feat2_384) # [B, 256]
            
            # Symmetric loss
            loss_qk, pos_qk, neg_qk = criterion(q, k)
            loss_kq, pos_kq, neg_kq = criterion(k, q)
            
            loss = (loss_qk + loss_kq) / 2
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pos_sim += (pos_qk + pos_kq) / 2
            train_neg_sim += (neg_qk + neg_kq) / 2
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if wandb.run:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/pos_sim": (pos_qk + pos_kq) / 2,
                    "train/neg_sim": (neg_qk + neg_kq) / 2,
                    "train/lr": optimizer.param_groups[0]["lr"]
                })
                
        train_loss /= len(train_loader)
        train_pos_sim /= len(train_loader)
        train_neg_sim /= len(train_loader)
        
        # Validation
        stgp_proj.eval()
        val_loss = 0.0
        val_pos_sim = 0.0
        val_neg_sim = 0.0
        correct_top1 = 0
        total_samples = 0
        
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for batch in pbar_val:
                img1 = batch["img1"].to(device)
                mask1 = batch["mask1"].to(device)
                img2 = batch["img2"].to(device)
                mask2 = batch["mask2"].to(device)
                
                out1 = dinov3(img1).last_hidden_state[:, -196:, :]
                out2 = dinov3(img2).last_hidden_state[:, -196:, :]
                
                feat1_384 = mask_pool_features(out1, mask1)
                feat2_384 = mask_pool_features(out2, mask2)
                
                q = stgp_proj(feat1_384)
                k = stgp_proj(feat2_384)
                
                loss_qk, pos_qk, neg_qk = criterion(q, k)
                loss_kq, pos_kq, neg_kq = criterion(k, q)
                
                loss = (loss_qk + loss_kq) / 2
                
                val_loss += loss.item()
                val_pos_sim += (pos_qk + pos_kq) / 2
                val_neg_sim += (neg_qk + neg_kq) / 2
                
                # Top-1 accuracy (is positive the most similar?)
                sim = torch.matmul(q, k.T)
                max_idx = sim.argmax(dim=1)
                correct_top1 += (max_idx == torch.arange(q.size(0), device=device)).sum().item()
                total_samples += q.size(0)
                
        val_loss /= len(val_loader)
        val_pos_sim /= len(val_loader)
        val_neg_sim /= len(val_loader)
        val_acc = correct_top1 / total_samples
        
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Pos Sim: {val_pos_sim:.4f} | Neg Sim: {val_neg_sim:.4f}")
        
        if wandb.run:
            wandb.log({
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/pos_sim": val_pos_sim,
                "val/neg_sim": val_neg_sim,
                "epoch": epoch
            })
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = checkpoints_dir / "best_stgp_proj.pt"
            torch.save(stgp_proj.state_dict(), ckpt_path)
            print(f"Saved new best checkpoint to {ckpt_path}")

    print("Training complete.")

if __name__ == "__main__":
    main()
