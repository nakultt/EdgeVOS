"""
Bundles EdgeVOS individual component weights into a single unified checkpoint.
"""
import torch
import os

def bundle():
    print("Bundling EdgeVOS weights into a single unified checkpoint...")
    
    # 1. Load the distillation checkpoint (Backbone + SpatialMaskHead)
    distill_path = "checkpoints/best_distill.pt"
    if not os.path.exists(distill_path):
        raise FileNotFoundError(f"Missing {distill_path}")
    
    distill_ckpt = torch.load(distill_path, map_location="cpu", weights_only=False)
    
    # Extract just the model state dict
    state_dict = distill_ckpt["model"]
    
    # 2. Load the STGP projection layer
    stgp_path = "checkpoints/best_stgp_proj.pt"
    if os.path.exists(stgp_path):
        stgp_state = torch.load(stgp_path, map_location="cpu", weights_only=True)
        # Prefix the stgp projection keys so they don't clash
        for k, v in stgp_state.items():
            state_dict[f"stgp_projection.{k}"] = v
        print(f"Added STGP projection weights ({len(stgp_state)} keys).")
    else:
        print(f"Warning: {stgp_path} not found. Skipping STGP weights.")

    # 3. Create the unified checkpoint dictionary
    unified_ckpt = {
        "version": "1.0",
        "model_name": "EdgeVOS",
        "mask_head_type": "SpatialMaskHead",
        "state_dict": state_dict,
        "config": {
            "dinov3_model": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "backbone": "repvit_m1",
            "mask_out_size": 256,
            "stgp_proj_dim": 256
        }
    }
    
    out_path = "checkpoints/edgevos_spatial_v1.pt"
    torch.save(unified_ckpt, out_path)
    print(f"\nSuccessfully bundled {len(state_dict)} tensors into {out_path}!")
    print(f"File size: {os.path.getsize(out_path) / (1024*1024):.1f} MB")

if __name__ == "__main__":
    bundle()
