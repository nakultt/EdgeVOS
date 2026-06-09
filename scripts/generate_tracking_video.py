"""
EdgeVOS — Generate tracking visualization video
"""

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.davis_dataset import DAVISDataset
from models.stgp import STGPProjection
from models.mask_head import SpatialMaskHead
from models.regrounding import RegroundingController
from models.backbone import BackboneWithRoI
from inference.pipeline import EdgeVOSPipeline

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    device = get_device()
    print(f"Using device: {device}")
    
    val_dataset = DAVISDataset(root=cfg.data.davis_root, split="val", resolution="480p")
    
    from transformers import AutoModel
    dinov3 = AutoModel.from_pretrained(cfg.model.dinov3_model, trust_remote_code=True).to(device).eval()
    for p in dinov3.parameters(): p.requires_grad = False
        
    backbone = BackboneWithRoI(model_name="repvit_m1").to(device).eval()
    mask_head = SpatialMaskHead(
        in_channels=backbone.feat_channels,  # 384
        out_size=cfg.model.mask_out_size
    ).to(device).eval()
    
    distill_ckpt = torch.load("checkpoints/best_distill.pt", map_location=device)
    bb_state = {k.replace("backbone.", "", 1): v for k, v in distill_ckpt["model"].items() if k.startswith("backbone.")}
    mh_state = {k.replace("mask_head.", "", 1): v for k, v in distill_ckpt["model"].items() if k.startswith("mask_head.")}
    backbone.load_state_dict(bb_state)
    mask_head.load_state_dict(mh_state)
    
    stgp_proj = STGPProjection(in_dim=cfg.model.dinov3_feat_dim, out_dim=cfg.model.stgp_proj_dim).to(device).eval()
    stgp_ckpt = torch.load("checkpoints/best_stgp_proj.pt", map_location=device)
    stgp_proj.load_state_dict(stgp_ckpt)
    
    regrounding = RegroundingController(model_name="MobileCLIP2-S2", pretrained="dfndr2b", device=device)
    
    class DummySAM:
        def predict(self, *args, **kwargs): return None
        
    pipeline = EdgeVOSPipeline(
        sam3_model=DummySAM(),
        dinov3_model=dinov3,
        mask_head=mask_head,
        stgp_projection=stgp_proj,
        regrounding=regrounding,
        cfg=cfg,
        device=device,
        backbone=backbone
    )
    
    target_seq = "dog" # Use 'dog' sequence for visualization
    seq = None
    for i in range(len(val_dataset)):
        if val_dataset[i]["name"] == target_seq:
            seq = val_dataset[i]
            break
            
    if seq is None:
        print(f"Sequence {target_seq} not found.")
        return

    frames = seq["frames"].to(device) # [T, 3, H, W]
    masks = seq["masks"].numpy() # [T, H, W]
    
    T = len(frames)
    H, W = frames.shape[2], frames.shape[3]
    
    # Initialize pipeline
    pipeline._frame_count = 0
    gt_mask_t0 = torch.from_numpy(masks[0]).to(device)
    y_indices, x_indices = torch.where(gt_mask_t0)
    if y_indices.numel() > 0:
        y1, y2 = y_indices.min().item(), y_indices.max().item() + 1
        x1, x2 = x_indices.min().item(), x_indices.max().item() + 1
        bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=device)
    else:
        bbox = torch.tensor([0.0, 0.0, float(W), float(H)], dtype=torch.float32, device=device)
        
    pipeline._last_bboxes = {0: bbox}
    feat_384 = pipeline._extract_dinov3_feat(frames[0], gt_mask_t0)
    pipeline.graph.init_object(0, feat_384, gt_mask_t0, frame_t=0)
    
    # Artifact directory path
    artifact_dir = r"C:\Users\nakul\.gemini\antigravity-ide\brain\2504cb4b-a7bf-4504-8175-6a1079c7ac10"
    out_path = os.path.join(artifact_dir, f"{target_seq}_tracking.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 15
    out_video = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    
    # Render frame 0
    f0_rgb = (frames[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    f0_rgb = cv2.cvtColor(f0_rgb, cv2.COLOR_RGB2BGR)
    mask0 = masks[0]
    overlay = f0_rgb.copy()
    overlay[mask0 == 1] = [0, 0, 255]
    f0_bgr = cv2.addWeighted(overlay, 0.5, f0_rgb, 0.5, 0)
    out_video.write(f0_bgr)
    
    for t in tqdm(range(1, T), desc="Generating Video"):
        with torch.no_grad(), torch.amp.autocast('cuda'):
            result = pipeline.process_frame(frames[t], obj_ids=[0], disable_regrounding=True)
        pred_mask = result[0]
        
        frame_rgb = (frames[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        if pred_mask is not None:
            pred_np = pred_mask.cpu().numpy().astype(bool)
            overlay = frame_bgr.copy()
            overlay[pred_np] = [0, 0, 255] 
            frame_bgr = cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0)
            
        out_video.write(frame_bgr)
        
    out_video.release()
    print(f"Video saved to {out_path}")

if __name__ == "__main__":
    main()
