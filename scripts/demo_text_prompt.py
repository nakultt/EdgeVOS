"""
EdgeVOS — Text-Prompted Video Object Tracking Demo
Takes a text prompt, locates the object in Frame 0 using MobileCLIP2-S2,
then tracks it across the video using STGP + SpatialMaskHead.
"""

import os
import argparse
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
from transformers import AutoModel
import open_clip
import torch.nn.functional as F
from PIL import Image

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def locate_object_by_text(frame_tensor, text_prompt, clip_model, preprocess, tokenizer, device):
    """
    Finds the bounding box of the text prompt in the frame using MobileCLIP2-S2.
    Uses a multi-scale sliding window approach for better localization than the 2x4 grid.
    """
    print(f"Locating '{text_prompt}' in Frame 0...")
    tokens = tokenizer([text_prompt]).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_feat = clip_model.encode_text(tokens)
        text_feat = F.normalize(text_feat, dim=-1)

    H, W = frame_tensor.shape[1], frame_tensor.shape[2]
    
    # Generate multi-scale crops
    crops = []
    bboxes = []
    
    # Scale 1: 2x2 grid
    for r in range(2):
        for c in range(2):
            y1, y2 = int(r*H/2), int((r+1)*H/2)
            x1, x2 = int(c*W/2), int((c+1)*W/2)
            crops.append(frame_tensor[:, y1:y2, x1:x2])
            bboxes.append([x1, y1, x2, y2])
            
    # Scale 2: 3x3 grid
    for r in range(3):
        for c in range(3):
            y1, y2 = int(r*H/3), int((r+1)*H/3)
            x1, x2 = int(c*W/3), int((c+1)*W/3)
            crops.append(frame_tensor[:, y1:y2, x1:x2])
            bboxes.append([x1, y1, x2, y2])
            
    # Scale 3: Central crops
    y1, y2 = int(H*0.25), int(H*0.75)
    x1, x2 = int(W*0.25), int(W*0.75)
    crops.append(frame_tensor[:, y1:y2, x1:x2])
    bboxes.append([x1, y1, x2, y2])
    
    # Scale 4: Full image
    crops.append(frame_tensor)
    bboxes.append([0, 0, W, H])

    scores = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for crop in crops:
            crop_np = crop.permute(1, 2, 0).cpu().numpy()
            crop_pil = Image.fromarray((crop_np * 255).astype(np.uint8))
            crop_input = preprocess(crop_pil).unsqueeze(0).to(device)
            img_feat = clip_model.encode_image(crop_input)
            img_feat = F.normalize(img_feat, dim=-1)
            score = (text_feat @ img_feat.T).item()
            scores.append(score)

    best_idx = np.argmax(scores)
    best_bbox = bboxes[best_idx]
    print(f"Found '{text_prompt}' at bbox {best_bbox} with score {scores[best_idx]:.3f}")
    return torch.tensor(best_bbox, dtype=torch.float32, device=device)

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    seq_name = cfg.get("seq", "dog")
    prompt = cfg.get("prompt", "dog")

    device = get_device()
    print(f"Tracking sequence: '{seq_name}' with prompt: '{prompt}' on {device}")
    
    val_dataset = DAVISDataset(root=cfg.data.davis_root, split="val", resolution="480p")
    seq = None
    for i in range(len(val_dataset)):
        if val_dataset[i]["name"] == seq_name:
            seq = val_dataset[i]
            break
            
    if seq is None:
        print(f"Sequence {seq_name} not found.")
        return

    # Load all models
    print("Loading models...")
    dinov3 = AutoModel.from_pretrained(cfg.model.dinov3_model, trust_remote_code=True).to(device).eval()
    for p in dinov3.parameters(): p.requires_grad = False
        
    backbone = BackboneWithRoI(model_name="repvit_m1").to(device).eval()
    mask_head = SpatialMaskHead(
        in_channels=backbone.feat_channels, 
        out_size=cfg.model.mask_out_size
    ).to(device).eval()
    
    # Load custom distilled checkpoints
    distill_ckpt = torch.load("checkpoints/best_distill.pt", map_location=device)
    bb_state = {k.replace("backbone.", "", 1): v for k, v in distill_ckpt["model"].items() if k.startswith("backbone.")}
    mh_state = {k.replace("mask_head.", "", 1): v for k, v in distill_ckpt["model"].items() if k.startswith("mask_head.")}
    backbone.load_state_dict(bb_state)
    mask_head.load_state_dict(mh_state)
    
    stgp_proj = STGPProjection(in_dim=cfg.model.dinov3_feat_dim, out_dim=cfg.model.stgp_proj_dim).to(device).eval()
    stgp_ckpt = torch.load("checkpoints/best_stgp_proj.pt", map_location=device)
    stgp_proj.load_state_dict(stgp_ckpt)
    
    regrounding = RegroundingController(model_name="MobileCLIP2-S2", pretrained="dfndr2b", device=device)
    regrounding.register_text_prompt(0, prompt) # Cache text for M4 tracking recovery
    
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

    frames = seq["frames"].to(device) # [T, 3, H, W]
    T = len(frames)
    H, W = frames.shape[2], frames.shape[3]
    
    # 1. Initialize from Text
    pipeline._frame_count = 0
    bbox0 = locate_object_by_text(
        frames[0], prompt, regrounding.clip_model, regrounding.preprocess, regrounding.tokenizer, device
    )
    pipeline._last_bboxes = {0: bbox0}
    
    # Generate exact mask from bbox using SpatialMaskHead
    print("Decoding exact mask from text-guided bbox...")
    # Dummy sims since we just use the bbox for decode
    dummy_sims = torch.zeros(196, device=device) 
    mask0 = pipeline._decode_mask(frames[0], bbox0, dummy_sims, H, W)
    
    # Init STGP Graph
    feat_384 = pipeline._extract_dinov3_feat(frames[0], mask0)
    pipeline.graph.init_object(0, feat_384, mask0, frame_t=0)
    
    # Video Setup
    artifact_dir = r"C:\Users\nakul\.gemini\antigravity-ide\brain\2504cb4b-a7bf-4504-8175-6a1079c7ac10"
    out_path = os.path.join(artifact_dir, f"{seq_name}_text_tracked_{prompt.replace(' ', '_')}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out_path, fourcc, 15, (W, H))
    
    # Render Frame 0
    f0_rgb = (frames[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    f0_bgr = cv2.cvtColor(f0_rgb, cv2.COLOR_RGB2BGR)
    overlay = f0_bgr.copy()
    overlay[mask0.cpu().numpy()] = [0, 255, 0]  # Green for frame 0 (text initialization)
    out_video.write(cv2.addWeighted(overlay, 0.5, f0_bgr, 0.5, 0))
    
    # 2. Track across video
    for t in tqdm(range(1, T), desc="Tracking"):
        with torch.no_grad(), torch.amp.autocast('cuda'):
            result = pipeline.process_frame(frames[t], obj_ids=[0])
        pred_mask = result[0]
        
        frame_rgb = (frames[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        if pred_mask is not None:
            pred_np = pred_mask.cpu().numpy().astype(bool)
            overlay = frame_bgr.copy()
            
            # Red for normal tracking, Blue if re-grounded
            color = [255, 0, 0] if pipeline._last_triggers.get(0) else [0, 0, 255]
            overlay[pred_np] = color 
            frame_bgr = cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0)
            
        out_video.write(frame_bgr)
        
    out_video.release()
    print(f"Saved text-prompted tracking video to: {out_path}")

if __name__ == "__main__":
    main()
