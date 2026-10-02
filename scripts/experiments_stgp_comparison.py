import os
import sys
import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
from transformers import AutoModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.backbone import BackboneWithRoI
from models.mask_head import MLPMaskHead
from models.stgp import STGPProjection, STGPGraph
from models.regrounding import RegroundingController
from inference.pipeline import EdgeVOSPipeline
from eval.davis_eval import evaluate_pipeline
from data.davis_dataset import DAVISDataset

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = DAVISDataset(root=cfg.data.davis_root, split=cfg.eval.davis_split)
    # Subset of first 3 sequences
    subset_dataset = torch.utils.data.Subset(val_dataset, [0, 1, 2])
    
    dinov3_model = AutoModel.from_pretrained(cfg.model.dinov3_model).to(device).eval()
    backbone = BackboneWithRoI(model_name=cfg.model.backbone, pretrained=False).to(device)
    mask_head = MLPMaskHead(
        in_dim=backbone.feat_channels,
        hidden_dims=cfg.model.mask_head_dims[1:],
        out_size=cfg.model.mask_out_size
    ).to(device)
    
    distill_ckpt = torch.load("checkpoints/best_distill.pt", map_location=device)
    bb_state = {k.replace("backbone.", "", 1): v for k, v in distill_ckpt["model"].items() if k.startswith("backbone.")}
    mh_state = {k.replace("mask_head.", "", 1): v for k, v in distill_ckpt["model"].items() if k.startswith("mask_head.")}
    backbone.load_state_dict(bb_state)
    mask_head.load_state_dict(mh_state)
    backbone.eval()
    mask_head.eval()
    
    regrounding = RegroundingController(device=device)
    
    # ── Option A: Projected Features ──
    print("\n--- Running Option A (Projected Features) ---")
    stgp_proj = STGPProjection(in_dim=cfg.model.dinov3_feat_dim, out_dim=cfg.model.stgp_proj_dim).to(device)
    stgp_ckpt = torch.load("checkpoints/best_stgp_proj.pt", map_location=device)
    stgp_proj.load_state_dict(stgp_ckpt)
    stgp_proj.eval()
    
    pipeline_a = EdgeVOSPipeline(
        sam3_model=None,
        dinov3_model=dinov3_model,
        mask_head=mask_head,
        stgp_projection=stgp_proj,
        regrounding=regrounding,
        cfg=cfg,
        device=device,
        backbone=backbone
    )
    evaluate_pipeline(pipeline_a, subset_dataset, device=device)
    
    # ── Option B: Raw DINOv3 Features (Bypass Projection) ──
    print("\n--- Running Option B (Raw DINOv3 Features) ---")
    # We bypass projection by overriding the stgp_proj forward to be identity normalized
    class IdentityProj(torch.nn.Module):
        def forward(self, x):
            return F.normalize(x, dim=-1)
            
    identity_proj = IdentityProj()
    pipeline_b = EdgeVOSPipeline(
        sam3_model=None,
        dinov3_model=dinov3_model,
        mask_head=mask_head,
        stgp_projection=identity_proj,
        regrounding=regrounding,
        cfg=cfg,
        device=device,
        backbone=backbone
    )
    # Need to update pipeline graph keyframe interval / dimensions since dim is 384 now
    pipeline_b.graph.proj = identity_proj
    evaluate_pipeline(pipeline_b, subset_dataset, device=device)

if __name__ == "__main__":
    main()
