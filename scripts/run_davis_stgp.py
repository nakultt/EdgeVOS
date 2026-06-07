"""
EdgeVOS — Month 3: DAVIS-2017 STGP Benchmark Pipeline
Loads trained components and evaluates across multiple K (keyframe interval) values.
"""

import os
import sys
import json
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModel
from prettytable import PrettyTable

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.backbone import BackboneWithRoI
from models.mask_head import MLPMaskHead
from models.stgp import STGPProjection
from models.regrounding import RegroundingController
from inference.pipeline import EdgeVOSPipeline
from eval.davis_eval import evaluate_pipeline
from data.davis_dataset import DAVISDataset


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("EdgeVOS — Running DAVIS-2017 STGP Benchmark")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Components
    print("\n[1/5] Loading DINOv3 ViT-S...")
    dinov3_model = AutoModel.from_pretrained(cfg.model.dinov3_model).to(device).eval()
    
    print("[2/5] Loading M2 Mask Head (from checkpoints/best_distill.pt)...")
    # Initialize the student architecture to cleanly load the weights
    backbone = BackboneWithRoI(model_name=cfg.model.backbone, pretrained=False).to(device)
    mask_head = MLPMaskHead(
        in_dim=backbone.feat_channels,
        hidden_dims=cfg.model.mask_head_dims[1:],
        out_size=cfg.model.mask_out_size
    ).to(device)
    
    distill_ckpt_path = "checkpoints/best_distill.pt"
    if not os.path.exists(distill_ckpt_path):
        raise FileNotFoundError(f"Missing {distill_ckpt_path}. Ensure distillation training is complete.")
        
    distill_ckpt = torch.load(distill_ckpt_path, map_location=device)
    # Extract sub-components from StudentModel state dict
    bb_state = {k.replace("backbone.", "", 1): v for k, v in distill_ckpt["model"].items() if k.startswith("backbone.")}
    mh_state = {k.replace("mask_head.", "", 1): v for k, v in distill_ckpt["model"].items() if k.startswith("mask_head.")}
    
    backbone.load_state_dict(bb_state)
    mask_head.load_state_dict(mh_state)
    backbone.eval()
    mask_head.eval()
    
    print("[3/5] Loading M3 STGP Projection (from checkpoints/best_stgp_proj.pt)...")
    stgp_proj = STGPProjection(
        in_dim=cfg.model.dinov3_feat_dim,
        out_dim=cfg.model.stgp_proj_dim
    ).to(device)
    
    stgp_ckpt_path = "checkpoints/best_stgp_proj.pt"
    if not os.path.exists(stgp_ckpt_path):
        raise FileNotFoundError(f"Missing {stgp_ckpt_path}. Ensure STGP contrastive training is complete.")
        
    stgp_ckpt = torch.load(stgp_ckpt_path, map_location=device)
    stgp_proj.load_state_dict(stgp_ckpt)
    stgp_proj.eval()
    
    print("[4/5] Loading M4 Regrounding Controller (MobileCLIP2-S2)...")
    regrounding = RegroundingController(device=device)
    
    # 2. Build Pipeline
    print("[5/5] Assembling EdgeVOS Pipeline...")
    pipeline = EdgeVOSPipeline(
        sam3_model=None,  # Not needed for semi-supervised evaluation
        dinov3_model=dinov3_model,
        mask_head=mask_head,
        stgp_projection=stgp_proj,
        regrounding=regrounding,
        cfg=cfg,
        device=device,
        backbone=backbone
    )
    
    # Load Dataset
    print("\nLoading DAVIS-2017 Validation Split...")
    val_dataset = DAVISDataset(root=cfg.data.davis_root, split=cfg.eval.davis_split)
    
    # 3. Ablation Study
    k_values = cfg.eval.benchmark_k_values
    results_summary = []
    k5_results = None
    
    os.makedirs("eval/results", exist_ok=True)
    
    print("\nStarting K-Value Ablation Benchmark...")
    for K in k_values:
        print("-" * 50)
        print(f"Evaluating STGP with K = {K}")
        pipeline.graph.K = K
        
        # Run evaluation
        results = evaluate_pipeline(pipeline, val_dataset, device=device)
        
        # Calculate memory (in MB)
        mem_mb = pipeline.graph.memory_bytes() / 1e6
        
        row = {
            "K": K,
            "J Mean": results["J_mean"],
            "F Mean": results["F_mean"],
            "J&F Mean": results["JF_mean"],
            "Memory MB": mem_mb
        }
        results_summary.append(row)
        
        print(f"-> K={K} | J&F: {row['J&F Mean']:.4f} | Memory: {mem_mb:.2f} MB")
        
        # 4. Save results for K=5
        if K == 5:
            k5_results = results
            k5_path = "eval/results/stgp_K5.json"
            with open(k5_path, "w") as f:
                json.dump(k5_results, f, indent=4)
            print(f"Saved full K=5 results to {k5_path}")
            
    # 5. Print Comparison Table
    print("\n" + "=" * 60)
    print("Ablation Results Summary")
    print("=" * 60)
    
    try:
        from prettytable import PrettyTable
        table = PrettyTable()
        table.field_names = ["Method", "K", "J Mean", "F Mean", "J&F Mean", "Memory (MB)"]
        for res in results_summary:
            table.add_row(["EdgeVOS-STGP", res["K"], f"{res['J Mean']:.4f}", f"{res['F Mean']:.4f}", f"{res['J&F Mean']:.4f}", f"{res['Memory MB']:.2f}"])
        print(table)
    except ImportError:
        # Fallback if prettytable isn't installed
        header = f"{'Method':<15} | {'K':<4} | {'J Mean':<8} | {'F Mean':<8} | {'J&F Mean':<8} | {'Memory (MB)':<12}"
        print(header)
        print("-" * len(header))
        for res in results_summary:
            print(f"EdgeVOS-STGP    | {res['K']:<4} | {res['J Mean']:.4f} | {res['F Mean']:.4f} | {res['J&F Mean']:.4f} | {res['Memory MB']:.2f}")


if __name__ == "__main__":
    main()
