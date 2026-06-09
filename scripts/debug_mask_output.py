"""Debug: check what the mask head actually outputs for a DAVIS frame crop."""
import os, sys, torch, numpy as np, cv2
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.davis_dataset import DAVISDataset
from models.backbone import BackboneWithRoI
from models.mask_head import MLPMaskHead
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset
    val_dataset = DAVISDataset(root=cfg.data.davis_root, split="val", resolution="480p")
    
    # Load models
    backbone = BackboneWithRoI(model_name="repvit_m1").to(device).eval()
    mask_head = MLPMaskHead(
        in_dim=cfg.model.mask_head_dims[0],
        hidden_dims=cfg.model.mask_head_dims[1:],
        out_size=cfg.model.mask_out_size
    ).to(device).eval()
    
    ckpt = torch.load("checkpoints/best_distill.pt", map_location=device)
    bb_state = {k.replace("backbone.", "", 1): v for k, v in ckpt["model"].items() if k.startswith("backbone.")}
    mh_state = {k.replace("mask_head.", "", 1): v for k, v in ckpt["model"].items() if k.startswith("mask_head.")}
    backbone.load_state_dict(bb_state)
    mask_head.load_state_dict(mh_state)
    
    # Get car-shadow sequence  
    seq = None
    for i in range(len(val_dataset)):
        if val_dataset[i]["name"] == "car-shadow":
            seq = val_dataset[i]
            break
    
    frames = seq["frames"].to(device)  # [T, 3, H, W]
    masks = seq["masks"].numpy()
    H, W = frames.shape[2], frames.shape[3]
    
    # Get GT bbox from frame 0 mask
    gt_mask = masks[0]
    ys, xs = np.where(gt_mask > 0)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    
    # Expand bbox by 15%
    bw, bh = x2 - x1, y2 - y1
    pad = max(bw, bh) * 0.15
    ex1 = max(0, int(x1 - pad))
    ey1 = max(0, int(y1 - pad))
    ex2 = min(W, int(x2 + pad))
    ey2 = min(H, int(y2 + pad))
    
    print(f"Frame size: {H}x{W}")
    print(f"GT bbox: [{x1},{y1},{x2},{y2}]")
    print(f"Expanded bbox: [{ex1},{ey1},{ex2},{ey2}]")
    
    artifact_dir = r"C:\Users\nakul\.gemini\antigravity-ide\brain\2504cb4b-a7bf-4504-8175-6a1079c7ac10"
    
    for t in [0, 5, 10, 20]:
        frame = frames[t]
        
        # Method A: Crop first, then backbone with [0,0,1,1]
        crop = frame[:, ey1:ey2, ex1:ex2]
        crop_224 = F.interpolate(crop.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        norm_bbox = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            roi_feat = backbone(crop_224, norm_bbox)  # [1, 384]
            logits = mask_head(roi_feat)              # [1, 1, 256, 256]
            probs = logits.sigmoid()
        
        # Stats
        print(f"\n--- Frame {t} ---")
        print(f"  roi_feat: min={roi_feat.min():.3f} max={roi_feat.max():.3f} mean={roi_feat.mean():.3f}")
        print(f"  logits:   min={logits.min():.3f} max={logits.max():.3f} mean={logits.mean():.3f}")
        print(f"  probs:    min={probs.min():.3f} max={probs.max():.3f} mean={probs.mean():.3f}")
        print(f"  % > 0.5:  {(probs > 0.5).float().mean():.3f}")
        
        # Save probability map as image
        prob_np = probs[0, 0].cpu().numpy()
        prob_vis = (prob_np * 255).astype(np.uint8)
        prob_vis = cv2.applyColorMap(prob_vis, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(artifact_dir, f"debug_prob_t{t}.png"), prob_vis)
        
        # Save the crop for reference
        crop_np = (crop_224[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        crop_bgr = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(artifact_dir, f"debug_crop_t{t}.png"), crop_bgr)
        
        # Save binary mask overlay on crop
        mask_binary = (prob_np > 0.5).astype(np.uint8)
        mask_vis = cv2.resize(mask_binary * 255, (224, 224), interpolation=cv2.INTER_NEAREST)
        overlay = crop_bgr.copy()
        overlay[mask_vis > 0] = [0, 0, 255]
        blended = cv2.addWeighted(overlay, 0.5, crop_bgr, 0.5, 0)
        cv2.imwrite(os.path.join(artifact_dir, f"debug_overlay_t{t}.png"), blended)
    
    print("\nDebug images saved.")

if __name__ == "__main__":
    main()
