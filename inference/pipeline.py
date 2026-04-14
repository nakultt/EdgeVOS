"""
EdgeVOS — Full Inference Pipeline
M1 (SAM 3.1 detector) → M2 (MLP mask head) → M3 (STGP) → M4 (re-grounding)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from PIL import Image
import numpy as np

# Internal modules
from models.stgp import STGPGraph, STGPProjection
from models.mask_head import MLPMaskHead
from models.regrounding import RegroundingController


class EdgeVOSPipeline:
    """
    Full EdgeVOS inference pipeline.
    Input:  text prompt (once) + video frames (streaming)
    Output: per-frame binary masks with persistent object IDs
    """

    def __init__(
        self,
        sam3_model,         # SAM 3.1 teacher model (M1 grounding)
        dinov3_model,       # DINOv3 ViT-S (M3 feature extraction)
        mask_head:          MLPMaskHead,
        stgp_projection:    STGPProjection,
        regrounding:        RegroundingController,
        cfg,                # OmegaConf config
        device:             str = "cuda",
    ):
        self.sam3    = sam3_model
        self.dinov3  = dinov3_model
        self.m2      = mask_head
        self.rg      = regrounding
        self.device  = device
        self.cfg     = cfg

        # Build fresh STGP graph for each video
        self.graph = STGPGraph(
            projection           = stgp_projection,
            keyframe_interval    = cfg.stgp.keyframe_interval,
            sim_trigger          = cfg.stgp.sim_threshold_trigger,
            sim_recover          = cfg.stgp.sim_threshold_recover,
            drift_frames         = cfg.stgp.drift_frames,
            regrounding_interval = cfg.stgp.regrounding_interval,
            topk_patches         = cfg.stgp.topk_patches,
            ghost_confirm        = cfg.stgp.ghost_confirm_frames,
        )
        self._last_bboxes: Dict[int, torch.Tensor] = {}
        self._frame_count = 0

    # ── Public API ─────────────────────────────────────────────────

    def init_from_text(
        self,
        frame:       torch.Tensor,   # [3, H, W]
        text_prompt: str,
        obj_id:      int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Frame 0: Use SAM 3.1 to locate and segment the object from text.
        Initialises STGP graph node for this object.
        Returns initial mask [H, W] or None if not found.
        """
        # M1: SAM 3.1 text grounding
        pil_frame = Image.fromarray(
            (frame.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        )
        with torch.no_grad():
            result = self.sam3.predict(pil_frame, text=text_prompt)

        if result is None or len(result['boxes']) == 0:
            return None

        bbox = result['boxes'][0]                # [4] xyxy
        mask = result['masks'][0].squeeze()      # [H, W]

        # Register text prompt for M4 re-grounding
        self.rg.register_text_prompt(obj_id, text_prompt)

        # Extract DINOv3 feature for STGP initialisation
        feat_384 = self._extract_dinov3_feat(frame, mask)
        self.graph.init_object(obj_id, feat_384, mask, frame_t=0)
        self._last_bboxes[obj_id] = bbox

        return mask

    def process_frame(
        self,
        frame:   torch.Tensor,   # [3, H, W]
        obj_ids: List[int],
    ) -> Dict[int, Optional[torch.Tensor]]:
        """
        Process one video frame for all tracked objects.
        Returns dict: obj_id -> binary mask [H, W] or None if lost.
        """
        self._frame_count += 1
        frame_t = self._frame_count
        H, W    = frame.shape[1], frame.shape[2]

        # Extract DINOv3 patch features for full frame (shared across all objects)
        frame_patches = self._extract_dinov3_patches(frame)   # [196, 384]

        results = {}
        for obj_id in obj_ids:
            # M3: STGP propagation
            sims, topk_idx, needs_rg = self.graph.propagate(
                obj_id, frame_patches, frame_t
            )

            if sims is None:
                results[obj_id] = None
                continue

            # Convert topk patches → bounding box estimate
            pred_bbox = self._patches_to_bbox(topk_idx, H, W)

            if needs_rg:
                # M4: Re-grounding
                best_bbox, best_crop, score = self.rg.reground(
                    obj_id, frame, self._last_bboxes.get(obj_id), frame_t
                )
                if best_bbox is not None:
                    roi_feat = self._extract_roi_feat(frame_patches, best_bbox, H, W)
                    new_mask = self.m2.predict_mask(
                        roi_feat.unsqueeze(0), (H, W)
                    ).squeeze()
                    feat_384 = self._extract_dinov3_feat(frame, new_mask)
                    self.graph.add_regrounding_node(
                        obj_id, feat_384, new_mask, frame_t, score
                    )
                    self._last_bboxes[obj_id] = best_bbox
                    results[obj_id] = new_mask
                else:
                    results[obj_id] = None
            else:
                # M2: Generate mask from predicted region
                roi_feat = self._extract_roi_feat(frame_patches, pred_bbox, H, W)
                mask = self.m2.predict_mask(
                    roi_feat.unsqueeze(0), (H, W)
                ).squeeze()
                self._last_bboxes[obj_id] = pred_bbox

                # Add keyframe node if applicable
                if frame_t % self.graph.K == 0:
                    feat_384 = self._extract_dinov3_feat(frame, mask)
                    self.graph._add_keyframe_node(
                        obj_id, frame_patches, mask, frame_t,
                        conf=sims.max().item()
                    )
                results[obj_id] = mask

        return results

    # ── Feature extraction helpers ─────────────────────────────────

    def _extract_dinov3_patches(self, frame: torch.Tensor) -> torch.Tensor:
        """Extract 196 patch tokens [196, 384] from DINOv3 for a full frame."""
        inp = F.interpolate(
            frame.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
        ).to(self.device)
        with torch.no_grad():
            out = self.dinov3(pixel_values=inp, output_hidden_states=True)
        return out.last_hidden_state[0, 1:]  # [196, 384] — remove CLS

    def _extract_dinov3_feat(
        self, frame: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean-pool DINOv3 patches inside mask region → [384] feature."""
        patches = self._extract_dinov3_patches(frame)           # [196, 384]
        grid_mask = F.interpolate(
            mask.float().unsqueeze(0).unsqueeze(0),
            size=(14, 14), mode='nearest'
        ).squeeze().flatten()                                    # [196]
        inside = patches[grid_mask > 0.5]
        return inside.mean(0) if inside.numel() > 0 else patches.mean(0)

    def _extract_roi_feat(
        self,
        patches:  torch.Tensor,    # [196, 384]
        bbox:     torch.Tensor,    # [4] xyxy
        H:        int,
        W:        int,
    ) -> torch.Tensor:
        """Extract mean feature for patches within a bounding box region."""
        x1, y1, x2, y2 = bbox.tolist()
        # Map bbox to 14x14 patch grid
        px1 = int(x1 / W * 14)
        py1 = int(y1 / H * 14)
        px2 = int(x2 / W * 14)
        py2 = int(y2 / H * 14)
        inside = []
        for r in range(py1, min(py2+1, 14)):
            for c in range(px1, min(px2+1, 14)):
                inside.append(patches[r*14 + c])
        if not inside:
            return patches.mean(0)
        return torch.stack(inside).mean(0)

    def _patches_to_bbox(
        self, topk_idx: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """Convert top-k patch indices to a bounding box [x1,y1,x2,y2]."""
        rows = topk_idx // 14
        cols = topk_idx % 14
        patch_h = H / 14
        patch_w = W / 14
        x1 = (cols.min().item() * patch_w)
        y1 = (rows.min().item() * patch_h)
        x2 = ((cols.max().item() + 1) * patch_w)
        y2 = ((rows.max().item() + 1) * patch_h)
        return torch.tensor([x1, y1, x2, y2])
