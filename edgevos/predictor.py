"""
EdgeVOS High-Level Predictor API
Provides a clean, unified interface for open-vocabulary video object segmentation.

    model = EdgeVOSPredictor("checkpoints/edgevos_spatial_v1.pt")
    mask0 = model.init_from_text(model.frame_to_tensor(frame0_bgr), "cat")
    mask  = model.process_frame(model.frame_to_tensor(frame_bgr))   # None = object absent
"""
import os
import sys
from typing import Optional, Union

import numpy as np
import torch
from transformers import AutoModel

# Ensure project root is in path so internal imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import BackboneWithRoI
from models.mask_head import SpatialMaskHead
from models.regrounding import RegroundingController
from models.grounding import SAM3Grounder, CLIPGrounder
from inference.tracker import EdgeVOSTracker, DEFAULT_TRACKER_CFG


class EdgeVOSPredictor:
    def __init__(self, checkpoint_path: str, device: str = None, use_sam3: bool = True,
                 tracker_cfg: Optional[dict] = None):
        """
        Initialize the unified EdgeVOS model from a bundled checkpoint.

        use_sam3:    ground text prompts with SAM 3 (accurate). Falls back to
                     MobileCLIP2 dense window scoring if SAM 3 cannot be loaded.
        tracker_cfg: overrides for ``inference.tracker.DEFAULT_TRACKER_CFG``.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading EdgeVOS on {self.device}...")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        state_dict = ckpt["state_dict"]

        print("  Loading DINOv3 backbone...")
        self.dinov3 = AutoModel.from_pretrained(
            cfg.get("dinov3_model", "facebook/dinov3-vits16-pretrain-lvd1689m")
        ).to(self.device).eval()
        for p in self.dinov3.parameters():
            p.requires_grad = False

        print("  Loading RepViT encoder + SpatialMaskHead...")
        self.backbone = BackboneWithRoI(model_name=cfg.get("backbone", "repvit_m1"), pretrained=False)
        self.mask_head = SpatialMaskHead(in_channels=self.backbone.feat_channels,
                                         out_size=cfg.get("mask_out_size", 256))
        self.backbone.load_state_dict({k[len("backbone."):]: v for k, v in state_dict.items()
                                       if k.startswith("backbone.")})
        self.mask_head.load_state_dict({k[len("mask_head."):]: v for k, v in state_dict.items()
                                        if k.startswith("mask_head.")})
        self.backbone.to(self.device).eval()
        self.mask_head.to(self.device).eval()

        self.grounder = None
        if use_sam3:
            try:
                print("  Loading SAM 3 (text grounding)...")
                self.grounder = SAM3Grounder(device=self.device)
            except Exception as e:
                print(f"  SAM 3 unavailable ({type(e).__name__}: {e}); using MobileCLIP2 grounding.")
        if self.grounder is None:
            print("  Loading MobileCLIP2-S2 (text grounding)...")
            rg = RegroundingController(model_name="MobileCLIP2-S2", pretrained="dfndr2b", device=self.device)
            self.grounder = CLIPGrounder(rg.clip_model, rg.preprocess, rg.tokenizer, device=self.device)
            # CLIP grounding costs seconds per call and gives no mask: re-ground
            # only on scene cuts / sustained low confidence, not periodically.
            tracker_cfg = {"reground_interval": 0, "reground_cooldown": 30, **(tracker_cfg or {})}

        self.tracker = EdgeVOSTracker(self.dinov3, self.backbone, self.mask_head, self.grounder,
                                      device=self.device, cfg=tracker_cfg)
        print("EdgeVOS initialization complete!")

    # ── Input helpers ──────────────────────────────────────────────

    def frame_to_tensor(self, frame: Union[np.ndarray, torch.Tensor], bgr: bool = True) -> torch.Tensor:
        """
        Convert a frame to the [3, H, W] RGB float tensor in [0, 1] the model expects.
        numpy HxWx3 uint8 frames are assumed to be OpenCV BGR unless ``bgr=False``.
        """
        if isinstance(frame, np.ndarray):
            if bgr:
                frame = frame[..., ::-1]
            t = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)
            return t.to(self.device).float() / 255.0
        t = frame.to(self.device)
        return t.float() / 255.0 if t.dtype == torch.uint8 else t.float()

    # ── Tracking API ───────────────────────────────────────────────

    def init_from_text(self, frame_tensor: torch.Tensor, text_prompt: str, obj_id: int = 0) -> Optional[torch.Tensor]:
        """
        Start a new video: ground ``text_prompt`` in this frame (resets all state).
        Returns the initial mask [H, W] (bool), or None if the object is not visible yet —
        keep calling ``process_frame``; the tracker re-tries grounding until it appears.
        """
        return self.tracker.init(self.frame_to_tensor(frame_tensor), text_prompt)

    def process_frame(self, frame_tensor: torch.Tensor, obj_id: int = 0) -> Optional[torch.Tensor]:
        """Track into the next frame. Returns mask [H, W] (bool), or None if the object is absent."""
        return self.tracker.step(self.frame_to_tensor(frame_tensor))

    def reset(self) -> None:
        self.tracker.reset()

    @property
    def last_event(self) -> Optional[str]:
        """What happened on the last frame: 'init', 'reground:<reason>', 'absent' or None."""
        return self.tracker.last_event
