"""
EdgeVOS — M4 Re-Grounding Ablation on Synthetic Long Videos
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from collections import defaultdict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Internal modules
from data.davis_dataset import DAVISDataset
from models.stgp import STGPGraph, STGPProjection
from models.mask_head import MLPMaskHead
from models.regrounding import RegroundingController
from models.backbone import BackboneWithRoI
from inference.pipeline import EdgeVOSPipeline
from eval.davis_eval import jaccard, f_measure

import open_clip
import timm

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def create_synthetic_video(seq, target_len=501):
    """Loop sequence forward-backward to create continuous long video."""
    frames = seq["frames"]
    masks = seq["masks"].numpy()
    
    out_frames = []
    out_masks = []
    curr_len = 0
    
    forward = True
    while curr_len < target_len:
        if forward:
            out_frames.append(frames)
            out_masks.append(masks)
        else:
            out_frames.append(torch.flip(frames, dims=[0]))
            out_masks.append(masks[::-1])
        
        curr_len += len(frames)
        forward = not forward
        
    final_frames = torch.cat(out_frames, dim=0)[:target_len]
    final_masks = np.concatenate(out_masks, axis=0)[:target_len]
    return final_frames, final_masks

def run_tracking(pipeline, frames, gt_masks, text_prompt, disable_regrounding=False):
    device = get_device()
    T = len(frames)
    frames = frames.to(device)
    gt_mask_t0 = torch.from_numpy(gt_masks[0]).to(device)
    
    # Reset pipeline state
    pipeline.graph = pipeline.graph.__class__(
        projection=pipeline.graph.proj,
        keyframe_interval=pipeline.graph.K,
        **{k: getattr(pipeline.graph, k) for k in
           ['sim_trigger','sim_recover','drift_frames',
            'regrounding_interval','topk_patches','ghost_confirm']}
    )
    pipeline._frame_count = 0
    pipeline.rg._text_cache = {}
    pipeline.rg.register_text_prompt(0, text_prompt)
    
    y_indices, x_indices = torch.where(gt_mask_t0)
    if y_indices.numel() > 0:
        y1, y2 = y_indices.min().item(), y_indices.max().item() + 1
        x1, x2 = x_indices.min().item(), x_indices.max().item() + 1
        bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=device)
    else:
        H_f, W_f = frames.shape[2], frames.shape[3]
        bbox = torch.tensor([0.0, 0.0, float(W_f), float(H_f)], dtype=torch.float32, device=device)
    
    pipeline._last_bboxes = {0: bbox}
    feat_384 = pipeline._extract_dinov3_feat(frames[0], gt_mask_t0)
    pipeline.graph.init_object(0, feat_384, gt_mask_t0, frame_t=0)
    
    metrics = {
        "j_f_history": [],
        "triggers": defaultdict(int),
        "id_switches": 0
    }
    
    metrics["j_f_history"].append(1.0) # Frame 0 is perfect
    lost_streak = 0
    
    for t in tqdm(range(1, T), desc=f"Tracking (disable_rg={disable_regrounding})", leave=False):
        result = pipeline.process_frame(frames[t], obj_ids=[0], disable_regrounding=disable_regrounding)
        pred_mask = result[0]
        trigger = pipeline._last_triggers.get(0)
        
        if trigger is not None:
            metrics["triggers"][trigger] += 1
            
        if pred_mask is None:
            jf = 0.0
        else:
            pred_np = pred_mask.cpu().numpy()
            gt_np = gt_masks[t].astype(bool)
            j = jaccard(pred_np, gt_np)
            f = f_measure(pred_np, gt_np)
            jf = (j + f) / 2.0
            
        metrics["j_f_history"].append(jf)
        
        # Identity switch tracking (J&F < 0.1 for consecutive frames)
        if jf < 0.1:
            lost_streak += 1
        else:
            if lost_streak >= 3:
                # Recovered, counts as a switch/loss event
                metrics["id_switches"] += 1
            lost_streak = 0
            
    if lost_streak >= 3:
        metrics["id_switches"] += 1
        
    return metrics

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    device = get_device()
    
    print("Loading datasets and models...")
    val_dataset = DAVISDataset(root=cfg.data.davis_root, split="val", resolution="480p")
    
    from transformers import AutoModel
    dinov3 = AutoModel.from_pretrained(cfg.model.dinov3_model, trust_remote_code=True).to(device).eval()
    for p in dinov3.parameters(): p.requires_grad = False
        
    backbone = BackboneWithRoI(model_name="repvit_m1").to(device).eval()
    mask_head = MLPMaskHead(
        in_dim=cfg.model.mask_head_dims[0], 
        hidden_dims=cfg.model.mask_head_dims[1:], 
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
    
    # Need dummy SAM3 for initialization interface, though we skip it by passing GT directly
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
    
    target_seqs = ["bear", "car-shadow", "surf"]
    
    os.makedirs("eval/plots", exist_ok=True)
    
    all_metrics = {"with_m4": [], "without_m4": []}
    
    for name in target_seqs:
        seq = None
        for i in range(len(val_dataset)):
            if val_dataset[i]["name"] == name:
                seq = val_dataset[i]
                break
        if seq is None: continue
            
        print(f"\n--- Testing on Synthetic Long Sequence: {name} ---")
        frames, masks = create_synthetic_video(seq, target_len=501)
        text_prompt = name.replace("-", " ")
        
        print("Running WITHOUT M4 Re-Grounding...")
        m_without = run_tracking(pipeline, frames, masks, text_prompt, disable_regrounding=True)
        
        print("Running WITH M4 Re-Grounding...")
        m_with = run_tracking(pipeline, frames, masks, text_prompt, disable_regrounding=False)
        
        all_metrics["without_m4"].append(m_without)
        all_metrics["with_m4"].append(m_with)
        
        print(f"[{name}] Without M4: identity switches={m_without['id_switches']}, J&F@500={m_without['j_f_history'][-1]:.4f}")
        print(f"[{name}] With M4: identity switches={m_with['id_switches']}, J&F@500={m_with['j_f_history'][-1]:.4f}")
        print(f"         Triggers -> {dict(m_with['triggers'])}")

    # Aggregate and Plot
    plt.figure(figsize=(10, 6))
    
    t_frames = list(range(501))
    
    avg_without = np.mean([m["j_f_history"] for m in all_metrics["without_m4"]], axis=0)
    avg_with = np.mean([m["j_f_history"] for m in all_metrics["with_m4"]], axis=0)
    
    plt.plot(t_frames, avg_without, label='Without M4 Re-Grounding', color='red', linewidth=2, linestyle='--')
    plt.plot(t_frames, avg_with, label='With M4 Re-Grounding (EdgeVOS)', color='green', linewidth=2)
    
    plt.title('Long-Video Tracking Degradation Analysis (Synthetic 500+ Frames)', fontsize=14)
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('J&F Mean Score', fontsize=12)
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    save_path = "eval/plots/regrounding_ablation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")

if __name__ == "__main__":
    main()
