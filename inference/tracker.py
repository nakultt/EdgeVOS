"""
EdgeVOS — Streaming tracker used by EdgeVOSPredictor.

Per frame:
  1. DINOv3 patch features, once, at aspect-preserving resolution (M3 features).
  2. Patch-level fg/bg classification against a gated appearance memory
     (permanent anchor from the grounded frame + FIFO of confident keyframes).
  3. Motion window around the previous box → connected component that best
     overlaps the previous location → coarse object region.
  4. SpatialMaskHead (M2) on a tight crop of that region, gated by the
     dilated coarse region → final mask.
  5. Re-grounding (M4) through the text grounder on scene cuts, sustained low
     confidence, periodically, and repeatedly while the object is absent.

``inference/pipeline.py`` (EdgeVOSPipeline) is kept unchanged for the DAVIS
research benchmarks; this module is the production path.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.grounding import Detection

DEFAULT_TRACKER_CFG = dict(
    dino_long_side=448,        # DINOv3 input long side (multiple of 16); 384 → −0.04 IoU, 320 → −0.14
    knn=5,                     # top-k neighbours for fg/bg patch scoring
    max_fg_bank=500,           # patch features kept per memory slot (1500/3000 gave no gain,
    max_bg_bank=1000,          # at 3x the kNN cost: 4.6 → 1.5 GFLOPs/frame)
    recent_keyframes=4,        # FIFO keyframes in addition to the anchor
    keyframe_interval=5,       # try to add memory every K frames (if confident)
    keyframe_min_conf=0.08,    # fg/bg margin required to write memory
    search_margin=0.75,        # motion window = prev box ± margin·max(w, h)
    low_conf=0.02,             # margin below which a frame counts as low-confidence
    low_conf_frames=3,         # consecutive low-confidence frames → re-ground
    reground_cooldown=10,      # min frames between confidence-triggered re-groundings
    reground_interval=30,      # periodic re-grounding (frames); 0 disables
    absent_retry=5,            # while absent, re-ground every N frames
    cut_gray_diff=18.0,        # scene cut: mean abs grey-level change (64x36 thumbnail)...
    cut_gray_ratio=3.0,        # ...that is also this many times the recent median change
    cut_patch_frac=0.015,      # or: fraction of DINOv3 patches whose feature flips (cos < 0.5)
    min_det_appearance=-0.05,  # reject re-ground candidates that look unlike the memory
    mask_threshold=0.5,
    box_refine_iters=3,        # mask-head re-crop iterations for box-only detections
)

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class FrameFeatures:
    feats: torch.Tensor   # [gh*gw, C] L2-normalised patch features
    gh: int
    gw: int


def _box_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    ix = (torch.min(a[2], b[2]) - torch.max(a[0], b[0])).clamp(min=0)
    iy = (torch.min(a[3], b[3]) - torch.max(a[1], b[1])).clamp(min=0)
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union.clamp(min=1e-6))


class PatchMemory:
    """Anchor (never evicted) + FIFO of recent confident keyframes."""

    def __init__(self, recent: int, max_fg: int, max_bg: int):
        self.max_fg, self.max_bg = max_fg, max_bg
        self.anchor_fg: Optional[torch.Tensor] = None
        self.anchor_bg: Optional[torch.Tensor] = None
        self.recent = deque(maxlen=recent)

    @staticmethod
    def _cap(x: torch.Tensor, n: int) -> torch.Tensor:
        if x.shape[0] <= n:
            return x
        idx = torch.randperm(x.shape[0], device=x.device)[:n]
        return x[idx]

    def set_anchor(self, fg: torch.Tensor, bg: torch.Tensor) -> None:
        self.anchor_fg = self._cap(fg, self.max_fg)
        self.anchor_bg = self._cap(bg, self.max_bg)
        self.recent.clear()

    def add(self, fg: torch.Tensor, bg: torch.Tensor) -> None:
        self.recent.append((self._cap(fg, self.max_fg // 2), self._cap(bg, self.max_bg // 2)))

    def banks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        fg = [self.anchor_fg] + [f for f, _ in self.recent]
        bg = [self.anchor_bg] + [b for _, b in self.recent]
        return torch.cat(fg), torch.cat(bg)


class EdgeVOSTracker:
    def __init__(self, dinov3, backbone, mask_head, grounder, device: str = "cuda", cfg: Optional[dict] = None):
        self.dinov3 = dinov3
        self.backbone = backbone
        self.mask_head = mask_head
        self.grounder = grounder
        self.device = device
        self.cfg = {**DEFAULT_TRACKER_CFG, **(cfg or {})}
        self.n_prefix = 1 + int(getattr(dinov3.config, "num_register_tokens", 0))
        self.patch = int(getattr(dinov3.config, "patch_size", 16))
        self.mean = _IMAGENET_MEAN.to(device)
        self.std = _IMAGENET_STD.to(device)
        self.reset()

    # ── State ──────────────────────────────────────────────────────

    def reset(self) -> None:
        c = self.cfg
        self.prompt: Optional[str] = None
        self.memory = PatchMemory(c["recent_keyframes"], c["max_fg_bank"], c["max_bg_bank"])
        self.frame_t = 0
        self.box: Optional[torch.Tensor] = None       # last box, absolute xyxy
        self.area_ema: Optional[float] = None
        self.low_conf_count = 0
        self.last_reground_t = 0
        self.absent = False
        self.pending_box: Optional[torch.Tensor] = None   # unconfirmed re-acquisition
        self.prev_thumb = None
        self.prev_feats: Optional[torch.Tensor] = None
        self.gray_diffs = deque(maxlen=30)
        self.last_conf = 0.0
        self.last_event: Optional[str] = None         # "init" | "reground:<reason>" | "absent" | None

    # ── Public API ─────────────────────────────────────────────────

    @torch.no_grad()
    def init(self, frame: torch.Tensor, prompt: str) -> Optional[torch.Tensor]:
        """frame: [3, H, W] RGB float in [0, 1] on device."""
        self.reset()
        self.prompt = prompt
        ff = self._features(frame)
        self._scene_cut(frame, ff)
        det = self._select_detection(frame, ff, use_memory=False)
        if det is None:
            self.absent = True
            self.last_event = "absent"
            return None
        mask = self._detection_mask(frame, det)
        self._commit(ff, mask, anchor=True)
        self.last_event = "init"
        return mask

    @torch.no_grad()
    def step(self, frame: torch.Tensor) -> Optional[torch.Tensor]:
        c = self.cfg
        self.frame_t += 1
        self.last_event = None

        ff = self._features(frame)
        scene_cut = self._scene_cut(frame, ff)

        # Nothing anchored yet (prompt never found) or object currently absent.
        if self.memory.anchor_fg is None or self.absent:
            if scene_cut or (self.frame_t - self.last_reground_t) >= c["absent_retry"]:
                return self._reground(frame, ff, "absent" if not scene_cut else "scene_cut")
            return None

        if scene_cut:
            return self._reground(frame, ff, "scene_cut")

        mask, conf = self._propagate(frame, ff)
        self.last_conf = conf

        self.low_conf_count = self.low_conf_count + 1 if (mask is None or conf < c["low_conf"]) else 0
        cooled_down = self.frame_t - self.last_reground_t >= c["reground_cooldown"]
        if self.low_conf_count >= c["low_conf_frames"] and cooled_down:
            return self._reground(frame, ff, "low_confidence", fallback=mask)
        if c["reground_interval"] and self.frame_t - self.last_reground_t >= c["reground_interval"]:
            return self._reground(frame, ff, "periodic", fallback=mask)

        if mask is not None:
            # Gate memory writes: confident frame with a plausible area (no memory poisoning).
            area_ok = self.area_ema is None or 0.5 <= float(mask.sum()) / self.area_ema <= 2.0
            write_memory = self.frame_t % c["keyframe_interval"] == 0 and conf >= c["keyframe_min_conf"] and area_ok
            self._commit(ff, mask, keyframe=write_memory)
        return mask

    # ── Propagation ────────────────────────────────────────────────

    def _propagate(self, frame: torch.Tensor, ff: FrameFeatures) -> Tuple[Optional[torch.Tensor], float]:
        c = self.cfg
        H, W = frame.shape[1:]
        score = self._patch_scores(ff.feats).view(ff.gh, ff.gw)        # fg − bg margin

        # Motion window around the previous box (grid coordinates).
        x1, y1, x2, y2 = self.box.tolist()
        m = c["search_margin"] * max(x2 - x1, y2 - y1)
        gx1 = max(0, int((x1 - m) / W * ff.gw))
        gy1 = max(0, int((y1 - m) / H * ff.gh))
        gx2 = min(ff.gw, int(np.ceil((x2 + m) / W * ff.gw)))
        gy2 = min(ff.gh, int(np.ceil((y2 + m) / H * ff.gh)))
        window = torch.zeros_like(score, dtype=torch.bool)
        window[gy1:gy2, gx1:gx2] = True

        fg = ((score > 0) & window).cpu().numpy().astype(np.uint8)
        n, labels = cv2.connectedComponents(fg, connectivity=8)
        if n <= 1:
            return None, float(score[window].max())

        # Component that overlaps the previous box most, tie-broken by score mass.
        prev = np.zeros_like(fg)
        prev[int(y1 / H * ff.gh):int(np.ceil(y2 / H * ff.gh)), int(x1 / W * ff.gw):int(np.ceil(x2 / W * ff.gw))] = 1
        score_np = score.float().cpu().numpy()
        best, best_key = 0, None
        for k in range(1, n):
            comp = labels == k
            key = ((comp & (prev > 0)).sum(), score_np[comp].clip(min=0).sum())
            if best_key is None or key > best_key:
                best, best_key = k, key
        comp = torch.from_numpy(labels == best).to(self.device)
        conf = float(score[comp].mean())

        # Coarse region in pixel coordinates.
        ys, xs = torch.where(comp)
        cell_w, cell_h = W / ff.gw, H / ff.gh
        coarse_box = torch.tensor([
            xs.min().item() * cell_w, ys.min().item() * cell_h,
            (xs.max().item() + 1) * cell_w, (ys.max().item() + 1) * cell_h,
        ], device=self.device)

        # Gate: coarse component dilated by one patch, upsampled to full res.
        gate = F.max_pool2d(comp.float()[None, None], 3, stride=1, padding=1)
        gate = F.interpolate(gate, size=(H, W), mode="bilinear", align_corners=False)[0, 0] > 0.5

        probs = self._mask_head_probs(frame, coarse_box)
        mask = (probs > c["mask_threshold"]) & gate
        if mask.sum() < 0.15 * gate.sum():
            # Mask head rejected most of the region (e.g. object partly outside
            # the crop) — fall back to the smooth patch-level mask.
            soft = F.interpolate(score[None, None].float(), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            mask = (soft > 0) & gate
        if mask.sum() == 0:
            return None, conf
        return mask, conf

    def _patch_scores(self, feats: torch.Tensor) -> torch.Tensor:
        fg_bank, bg_bank = self.memory.banks()
        k = self.cfg["knn"]
        s_fg = (feats @ fg_bank.T).topk(min(k, fg_bank.shape[0]), dim=1).values.mean(1)
        s_bg = (feats @ bg_bank.T).topk(min(k, bg_bank.shape[0]), dim=1).values.mean(1)
        return s_fg - s_bg

    # ── Re-grounding (M4) ──────────────────────────────────────────

    def _reground(self, frame, ff, reason: str, fallback: Optional[torch.Tensor] = None):
        self.last_reground_t = self.frame_t
        det = self._select_detection(frame, ff, use_memory=self.memory.anchor_fg is not None)
        if det is None:
            if fallback is not None and reason == "periodic":
                # Grounder missed but tracking is confident — keep tracking.
                self._commit(ff, fallback, keyframe=False)
                return fallback
            self.absent = True
            self.low_conf_count = 0
            self.pending_box = None
            self.last_event = "absent"
            return None
        # Re-acquiring after absence with a box-only (CLIP) grounder: require the
        # detection to repeat at the same place on the next retry, since single
        # false hits are common for semantically close prompts.
        acquiring = self.absent or self.memory.anchor_fg is None
        if acquiring and det.box is not None and not getattr(self.grounder, "provides_masks", True):
            pending, self.pending_box = self.pending_box, det.box
            if pending is None or _box_iou(pending, det.box) < 0.3:
                self.absent = True
                self.last_event = "absent"
                return None
        self.pending_box = None
        mask = self._detection_mask(frame, det)
        first = self.memory.anchor_fg is None
        self._commit(ff, mask, anchor=first, keyframe=not first)
        self.absent = False
        self.low_conf_count = 0
        self.last_event = f"reground:{reason}"
        return mask

    def _select_detection(self, frame, ff: FrameFeatures, use_memory: bool) -> Optional[Detection]:
        dets = self.grounder.detect(frame, self.prompt)
        if not dets:
            return None
        if not use_memory:
            return dets[0]
        score = self._patch_scores(ff.feats).view(ff.gh, ff.gw)
        best, best_val = None, -1e9
        for d in dets:
            # Judge appearance on the object mask, not a loose box full of background.
            d.mask = self._detection_mask(frame, d)
            m = d.mask
            grid = self._grid_fraction(m, ff) > 0.5
            appearance = float(score[grid].mean()) if grid.any() else -1.0
            if appearance < self.cfg["min_det_appearance"] and d.score < 0.6:
                continue
            val = d.score + 4.0 * appearance
            if val > best_val:
                best, best_val = d, val
        return best

    def _detection_mask(self, frame, det: Detection) -> torch.Tensor:
        if det.mask is not None and det.mask.any():
            return det.mask.bool()
        # Box-only grounder: let the distilled mask head segment the object and
        # re-crop on its own mask, so a coarse box converges to the tight
        # object box the head was trained on.
        H, W = frame.shape[1:]
        box, mask = det.box, None
        for _ in range(self.cfg["box_refine_iters"]):
            m = (self._mask_head_probs(frame, box) > self.cfg["mask_threshold"]) & self._box_mask(box, H, W, pad=0.15)
            if not m.any():
                break
            mask = m
            ys, xs = torch.where(m)
            box = torch.tensor([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=torch.float32, device=self.device)
        return mask if mask is not None else self._box_mask(det.box, H, W)

    # ── Memory / state update ──────────────────────────────────────

    def _commit(self, ff: FrameFeatures, mask: torch.Tensor, anchor: bool = False, keyframe: bool = False) -> None:
        if not mask.any():
            return
        frac = self._grid_fraction(mask, ff).flatten()
        if anchor or keyframe:
            fg = ff.feats[frac > 0.5]
            if fg.shape[0] == 0:
                fg = ff.feats[frac.argmax()].unsqueeze(0)
            bg = ff.feats[frac < 0.05]
            if anchor:
                self.memory.set_anchor(fg, bg)
            else:
                self.memory.add(fg, bg)
        ys, xs = torch.where(mask)
        self.box = torch.tensor([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                                dtype=torch.float32, device=self.device)
        area = float(mask.sum())
        self.area_ema = area if self.area_ema is None else 0.8 * self.area_ema + 0.2 * area

    # ── Feature helpers ────────────────────────────────────────────

    def _features(self, frame: torch.Tensor) -> FrameFeatures:
        H, W = frame.shape[1:]
        L = self.cfg["dino_long_side"]
        s = L / max(H, W)
        th = max(self.patch, int(round(H * s / self.patch)) * self.patch)
        tw = max(self.patch, int(round(W * s / self.patch)) * self.patch)
        x = F.interpolate(frame[None].float(), size=(th, tw), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        # fp32 on purpose: at batch size 1 these models are launch-bound and
        # autocast only adds cast kernels (measured ~1.5x slower).
        out = self.dinov3(pixel_values=x).last_hidden_state
        gh, gw = th // self.patch, tw // self.patch
        feats = F.normalize(out[0, self.n_prefix:self.n_prefix + gh * gw].float(), dim=-1)
        return FrameFeatures(feats=feats, gh=gh, gw=gw)

    @staticmethod
    def _grid_fraction(mask: torch.Tensor, ff: FrameFeatures) -> torch.Tensor:
        """Fraction of each patch cell covered by the mask → [gh, gw]."""
        return F.adaptive_avg_pool2d(mask.float()[None, None], (ff.gh, ff.gw))[0, 0]

    def _mask_head_probs(self, frame: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """SpatialMaskHead probabilities pasted into a full-frame [H, W] map."""
        H, W = frame.shape[1:]
        x1, y1, x2, y2 = box.tolist()
        pad = max(x2 - x1, y2 - y1, 1.0) * 0.15          # same context as distillation
        ex1, ey1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
        ex2, ey2 = min(W, int(np.ceil(x2 + pad))), min(H, int(np.ceil(y2 + pad)))
        full = torch.zeros(H, W, device=self.device)
        if ex2 - ex1 < 2 or ey2 - ey1 < 2:
            return full
        crop = F.interpolate(frame[None, :, ey1:ey2, ex1:ex2].float(), size=(224, 224),
                             mode="bilinear", align_corners=False)
        logits = self.mask_head(self.backbone._extract_feature_map(crop))
        probs = F.interpolate(logits.float().sigmoid(), size=(ey2 - ey1, ex2 - ex1),
                              mode="bilinear", align_corners=False)[0, 0]
        full[ey1:ey2, ex1:ex2] = probs
        return full

    def _box_mask(self, box: torch.Tensor, H: int, W: int, pad: float = 0.0) -> torch.Tensor:
        x1, y1, x2, y2 = box.tolist()
        p = max(x2 - x1, y2 - y1) * pad
        m = torch.zeros(H, W, dtype=torch.bool, device=self.device)
        m[max(0, int(y1 - p)):min(H, int(np.ceil(y2 + p))), max(0, int(x1 - p)):min(W, int(np.ceil(x2 + p)))] = True
        return m

    def _scene_cut(self, frame: torch.Tensor, ff: FrameFeatures) -> bool:
        """Hard cut detector: abrupt grey-level change or many DINOv3 patches flipping."""
        c = self.cfg
        thumb = F.interpolate(frame[None].float().mean(1, keepdim=True), size=(36, 64), mode="area")[0, 0] * 255
        cut = False
        if self.prev_thumb is not None:
            diff = float((thumb - self.prev_thumb).abs().mean())
            median = float(np.median(self.gray_diffs)) if self.gray_diffs else 0.0
            cut = diff > c["cut_gray_diff"] and diff > c["cut_gray_ratio"] * median
            if self.prev_feats is not None and self.prev_feats.shape == ff.feats.shape:
                flipped = float(((ff.feats * self.prev_feats).sum(1) < 0.5).float().mean())
                cut = cut or flipped > c["cut_patch_frac"]
            if not cut:
                self.gray_diffs.append(diff)
        self.prev_thumb, self.prev_feats = thumb, ff.feats
        return cut
