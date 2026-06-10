"""
EdgeVOS High-Level Predictor API
Provides a clean, unified interface for open-vocabulary video object segmentation.
"""
import torch
import numpy as np
from transformers import AutoModel
import sys
import os

# Ensure project root is in path so internal imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import BackboneWithRoI
from models.mask_head import SpatialMaskHead
from models.stgp import STGPProjection
from models.regrounding import RegroundingController
from inference.pipeline import EdgeVOSPipeline

# Fallback config
from omegaconf import OmegaConf

class EdgeVOSPredictor:
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Initialize the unified EdgeVOS model from a bundled checkpoint.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading EdgeVOS on {self.device}...")
        
        # Load the bundled checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        state_dict = ckpt["state_dict"]
        
        # 1. Initialize Sub-Models
        print("  Loading DINOv3 backbone...")
        self.dinov3 = AutoModel.from_pretrained(
            cfg.get("dinov3_model", "facebook/dinov3-vits16-pretrain-lvd1689m"), 
            trust_remote_code=True
        ).to(self.device).eval()
        for p in self.dinov3.parameters(): p.requires_grad = False
            
        print("  Loading RepViT encoder...")
        self.backbone = BackboneWithRoI(
            model_name=cfg.get("backbone", "repvit_m1"), pretrained=False
        ).to(self.device).eval()
        
        print("  Loading SpatialMaskHead...")
        self.mask_head = SpatialMaskHead(
            in_channels=self.backbone.feat_channels, 
            out_size=cfg.get("mask_out_size", 256)
        ).to(self.device).eval()
        
        print("  Loading STGP Projection...")
        self.stgp_proj = STGPProjection(
            in_dim=self.dinov3.config.hidden_size if hasattr(self.dinov3.config, 'hidden_size') else 384,
            out_dim=cfg.get("stgp_proj_dim", 256)
        ).to(self.device).eval()
        
        print("  Loading MobileCLIP2-S2 (Text Regrounding)...")
        self.regrounding = RegroundingController(
            model_name="MobileCLIP2-S2", pretrained="dfndr2b", device=self.device
        )
        
        # 2. Distribute bundled weights to sub-models
        print("  Distributing weights...")
        bb_state = {k.replace("backbone.", "", 1): v for k, v in state_dict.items() if k.startswith("backbone.")}
        mh_state = {k.replace("mask_head.", "", 1): v for k, v in state_dict.items() if k.startswith("mask_head.")}
        stgp_state = {k.replace("stgp_projection.", "", 1): v for k, v in state_dict.items() if k.startswith("stgp_projection.")}
        
        self.backbone.load_state_dict(bb_state)
        self.mask_head.load_state_dict(mh_state)
        if stgp_state:
            self.stgp_proj.load_state_dict(stgp_state)
            
        # 3. Create underlying pipeline
        dummy_cfg = OmegaConf.create({
            "model": {"dinov3_feat_dim": 384, "stgp_proj_dim": 256, "mask_out_size": 256},
            "stgp": {
                "keyframe_interval": 5,
                "sim_threshold_trigger": 0.45,
                "sim_threshold_recover": 0.55,
                "drift_frames": 3,
                "regrounding_interval": 150,
                "topk_patches": 15,
                "ghost_confirm_frames": 5
            }
        })
        
        class DummySAM:
            def predict(self, *args, **kwargs): return None
            
        self.pipeline = EdgeVOSPipeline(
            sam3_model=DummySAM(),
            dinov3_model=self.dinov3,
            mask_head=self.mask_head,
            stgp_projection=self.stgp_proj,
            regrounding=self.regrounding,
            cfg=dummy_cfg,
            device=self.device,
            backbone=self.backbone
        )
        print("EdgeVOS initialization complete!")

    def set_text_prompt(self, text_prompt: str, obj_id: int = 0):
        """Register the text prompt for tracking recovery."""
        self.regrounding.register_text_prompt(obj_id, text_prompt)
        
    def init_from_text(self, frame_tensor: torch.Tensor, text_prompt: str, obj_id: int = 0) -> torch.Tensor:
        """
        Scan frame 0 for the object matching the text prompt, generate initial mask, and initialize STGP.
        Returns the initial mask tensor [H, W].
        """
        self.set_text_prompt(text_prompt, obj_id)
        
        # Borrow the logic from our demo script to locate the bbox
        import torch.nn.functional as F
        from PIL import Image
        
        tokens = self.regrounding.tokenizer([text_prompt]).to(self.device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            text_feat = self.regrounding.clip_model.encode_text(tokens)
            text_feat = F.normalize(text_feat, dim=-1)

        H, W = frame_tensor.shape[1], frame_tensor.shape[2]
        crops, bboxes = [], []
        
        # Basic multi-scale crops
        for r in range(2):
            for c in range(2):
                y1, y2 = int(r*H/2), int((r+1)*H/2)
                x1, x2 = int(c*W/2), int((c+1)*W/2)
                crops.append(frame_tensor[:, y1:y2, x1:x2])
                bboxes.append([x1, y1, x2, y2])
        crops.append(frame_tensor)
        bboxes.append([0, 0, W, H])

        scores = []
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for crop in crops:
                crop_np = crop.permute(1, 2, 0).cpu().numpy()
                crop_pil = Image.fromarray((crop_np * 255).astype(np.uint8))
                crop_input = self.regrounding.preprocess(crop_pil).unsqueeze(0).to(self.device)
                img_feat = self.regrounding.clip_model.encode_image(crop_input)
                img_feat = F.normalize(img_feat, dim=-1)
                scores.append((text_feat @ img_feat.T).item())

        best_idx = np.argmax(scores)
        best_bbox = torch.tensor(bboxes[best_idx], dtype=torch.float32, device=self.device)
        
        # Reset pipeline state
        self.pipeline._frame_count = 0
        self.pipeline._last_bboxes = {obj_id: best_bbox}
        
        # Decode initial mask
        dummy_sims = torch.zeros(196, device=self.device) 
        mask = self.pipeline._decode_mask(frame_tensor, best_bbox, dummy_sims, H, W)
        
        # Init graph
        feat_384 = self.pipeline._extract_dinov3_feat(frame_tensor, mask)
        self.pipeline.graph.init_object(obj_id, feat_384, mask, frame_t=0)
        
        return mask

    def process_frame(self, frame_tensor: torch.Tensor, obj_id: int = 0) -> torch.Tensor:
        """
        Process the next video frame and return the tracked mask.
        """
        with torch.no_grad(), torch.amp.autocast('cuda'):
            result = self.pipeline.process_frame(frame_tensor, obj_ids=[obj_id])
        return result[obj_id]
