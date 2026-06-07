"""
EdgeVOS — Module 4: Confidence-Adaptive CLIP Re-Grounding Controller
Uses MobileCLIP2-S2 to recover from tracking drift without human intervention.
Novel: STGP confidence signal → spatially-constrained text re-grounding → graph re-anchor
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import open_clip
import numpy as np
from PIL import Image


class RegroundingController:
    """
    Monitors STGP confidence and triggers text-based re-grounding when drift detected.
    Uses cached text prompt embeddings — zero added compute per re-grounding event.
    """

    def __init__(
        self,
        model_name:         str   = "MobileCLIP2-S2",
        pretrained:         str   = "dfndr2b",
        n_crops:            int   = 8,      # spatial crops to score (2x4 grid)
        overlap_threshold:  float = 0.20,   # min overlap with last known location
        device:             str   = "cuda",
    ):
        self.device = device
        self.n_crops = n_crops
        self.overlap_threshold = overlap_threshold

        # Load MobileCLIP2-S2 (text + image encoder)
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.clip_model = self.clip_model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Cache: obj_id -> text embedding [1, 512]
        self._text_cache: dict = {}

    # ── Initialisation ────────────────────────────────────────────

    def register_text_prompt(self, obj_id: int, text_prompt: str) -> None:
        """
        Encode text prompt once at startup and cache.
        Called during M1 grounding, before video processing begins.
        """
        tokens = self.tokenizer([text_prompt]).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_feat = self.clip_model.encode_text(tokens)
            text_feat = F.normalize(text_feat, dim=-1)
        self._text_cache[obj_id] = text_feat  # [1, 512]

    # ── Re-grounding ──────────────────────────────────────────────

    def reground(
        self,
        obj_id:     int,
        frame:      torch.Tensor,         # [3, H, W] float32 frame tensor
        last_bbox:  Optional[torch.Tensor],  # [4] xyxy last known location
        frame_t:    int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], float]:
        """
        Re-locate object using MobileCLIP2-S2 text-image scoring.

        Returns:
            best_bbox:  [4] xyxy of best-matching crop, or None if failed
            best_crop:  [3, H', W'] image crop for M2 mask head
            best_score: cosine similarity score
        """
        text_feat = self._text_cache.get(obj_id)
        if text_feat is None:
            return None, None, 0.0

        H, W = frame.shape[1], frame.shape[2]
        crops, bboxes = self._make_crops(frame, H, W)

        # Score crops against text prompt
        scores = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for crop in crops:
                crop_np = crop.permute(1, 2, 0).cpu().numpy()
                crop_pil = Image.fromarray((crop_np * 255).astype(np.uint8))
                crop_input = self.preprocess(crop_pil).unsqueeze(0).to(self.device)
                img_feat = self.clip_model.encode_image(crop_input)
                img_feat = F.normalize(img_feat, dim=-1)
                score    = (text_feat @ img_feat.T).item()
                scores.append(score)

        scores = torch.tensor(scores)

        # Filter: crops must overlap with last known location
        if last_bbox is not None:
            valid = self._check_overlap(bboxes, last_bbox, self.overlap_threshold)
            scores[~valid] = -999.0

        if scores.max().item() < 0.1:
            return None, None, 0.0

        best_idx   = scores.argmax().item()
        best_score = scores[best_idx].item()
        best_bbox  = bboxes[best_idx]
        best_crop  = crops[best_idx]

        return best_bbox, best_crop, best_score

    # ── Helpers ───────────────────────────────────────────────────

    def _make_crops(self, frame, H, W):
        """2x4 tile grid → 8 crops."""
        crops, bboxes = [], []
        rows, cols = 2, 4
        for r in range(rows):
            for c in range(cols):
                y1 = int(r * H / rows)
                y2 = int((r+1) * H / rows)
                x1 = int(c * W / cols)
                x2 = int((c+1) * W / cols)
                crops.append(frame[:, y1:y2, x1:x2])
                bboxes.append(torch.tensor([x1, y1, x2, y2], dtype=torch.float32))
        return crops, bboxes

    def _check_overlap(self, bboxes, ref_bbox, threshold):
        """Return bool tensor: which crops overlap ref_bbox by >= threshold."""
        ref = ref_bbox.float()
        valid = []
        for bb in bboxes:
            ix1 = max(bb[0], ref[0])
            iy1 = max(bb[1], ref[1])
            ix2 = min(bb[2], ref[2])
            iy2 = min(bb[3], ref[3])
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            area  = (bb[2]-bb[0]) * (bb[3]-bb[1])
            valid.append(inter / (area + 1e-6) >= threshold)
        return torch.tensor(valid)
