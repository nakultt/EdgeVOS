"""
EdgeVOS mobile runtime — the SAM-free product pipeline on ONNX Runtime + numpy.

No torch / transformers / open_clip / timm. Image resizing and connected
components use OpenCV, which is available on Android and iOS (or maps to the
platform's image APIs). This file is the reference that the Kotlin / Swift /
C++ app ports line by line; it mirrors ``inference/tracker.py``.

    from mobile.runtime import MobileTracker
    tracker = MobileTracker("mobile/models")
    mask = tracker.init(frame_rgb_uint8, "cat")   # [H, W] bool, or None
    mask = tracker.step(next_frame_rgb_uint8)     # None while the object is absent

Models come from ``python scripts/export_mobile.py``.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from mobile.tokenizer import CLIPTokenizer


# ── Small numpy helpers (each maps to one call on device) ──────────────────

def resize(img: np.ndarray, h: int, w: int, area: bool = False) -> np.ndarray:
    """Bilinear (align_corners=False) or area resize of an HxW or HxWxC float array."""
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if area else cv2.INTER_LINEAR)


def to_nchw(img_hwc: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img_hwc.transpose(2, 0, 1)[None], dtype=np.float32)


def pool_matrix(in_size: int, out_size: int) -> np.ndarray:
    """Row i averages the input bin adaptive average pooling assigns to output i."""
    m = np.zeros((out_size, in_size), np.float32)
    for i in range(out_size):
        s, e = (i * in_size) // out_size, -((-(i + 1) * in_size) // out_size)
        m[i, s:e] = 1.0 / (e - s)
    return m


def grid_fraction(mask: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """Fraction of each patch cell covered by the mask → [gh, gw]."""
    H, W = mask.shape
    return pool_matrix(H, gh) @ mask.astype(np.float32) @ pool_matrix(W, gw).T


def bbox(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


def box_mask(box, H, W, pad=0.0) -> np.ndarray:
    x1, y1, x2, y2 = box
    p = max(x2 - x1, y2 - y1) * pad
    m = np.zeros((H, W), bool)
    m[max(0, int(y1 - p)):min(H, int(np.ceil(y2 + p))), max(0, int(x1 - p)):min(W, int(np.ceil(x2 + p)))] = True
    return m


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-6)


def topk_mean(sims: np.ndarray, k: int) -> np.ndarray:
    k = min(k, sims.shape[1])
    return np.partition(sims, -k, axis=1)[:, -k:].mean(1)


# ── Models ─────────────────────────────────────────────────────────────────

class Models:
    """ONNX Runtime sessions for every exported model (see scripts/export_mobile.py)."""

    def __init__(self, model_dir: str, providers: Optional[List[str]] = None, int8: bool = False):
        with open(os.path.join(model_dir, "manifest.json")) as f:
            self.manifest = json.load(f)
        providers = providers or ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        # Models run back-to-back with numpy work in between; spinning worker
        # threads would contend with it (measured 4x slower DINOv3 on CPU).
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")

        def session(name):
            info = self.manifest["models"][name]
            file = info["int8_file"] if int8 and "int8_file" in info else info["file"]
            return ort.InferenceSession(os.path.join(model_dir, file), options, providers=providers)

        self.dino = session("dinov3_patches")
        self.segmenter = session("segmenter")
        self.clip_dense = session("clip_dense")
        self.clip_text = session("clip_text")
        self.clip_background = np.load(os.path.join(model_dir, "clip_background.npy"))
        self.tokenizer = CLIPTokenizer()
        self.grounding = self.manifest["grounding"]

    def input_size(self, name: str, H: int, W: int) -> Tuple[int, int]:
        """Aspect-preserving input size: long side from the manifest, multiples of the stride."""
        info = self.manifest["models"][name]
        stride, s = info["stride"], info["long_side"] / max(H, W)
        return (max(stride, int(round(H * s / stride)) * stride),
                max(stride, int(round(W * s / stride)) * stride))

    def dino_patches(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        h, w = self.input_size("dinov3_patches", *frame.shape[:2])
        patches = self.dino.run(None, {"image": to_nchw(resize(frame, h, w))})[0]
        return patches, h // 16, w // 16

    def segment(self, crop: np.ndarray) -> np.ndarray:
        return self.segmenter.run(None, {"crop": to_nchw(resize(crop, 224, 224))})[0][0, 0]

    def dense_clip(self, frame: np.ndarray) -> np.ndarray:
        h, w = self.input_size("clip_dense", *frame.shape[:2])
        return self.clip_dense.run(None, {"image": to_nchw(resize(frame, h, w))})[0]

    def text(self, prompt: str) -> np.ndarray:
        return self.clip_text.run(None, {"tokens": self.tokenizer(prompt)})[0]


# ── Grounding (M1, SAM-free) ───────────────────────────────────────────────

@dataclass
class Detection:
    box: np.ndarray
    score: float
    mask: Optional[np.ndarray] = None


class Grounder:
    """Dense MobileCLIP2 heatmap → candidate boxes (mirrors models.grounding.CLIPGrounder)."""

    def __init__(self, models: Models, max_candidates: int = 3):
        self.m = models
        self.presence_threshold = models.grounding["presence_threshold"]
        self.template = models.grounding["prompt_template"]
        self.max_candidates = max_candidates
        self._text = {}

    def detect(self, frame: np.ndarray, prompt: str) -> List[Detection]:
        H, W = frame.shape[:2]
        if prompt not in self._text:
            self._text[prompt] = self.m.text(self.template.format(prompt))
        dense = self.m.dense_clip(frame)                                   # [h, w, 512]
        hm = (dense @ self._text[prompt].T)[..., 0] - (dense @ self.m.clip_background.T).max(-1)
        if hm.max() < self.presence_threshold:
            return []
        up = resize(hm.astype(np.float32), H, W)
        lo, hi = float(up.min()), float(up.max())
        n, labels = cv2.connectedComponents((up >= lo + 0.6 * (hi - lo)).astype(np.uint8), connectivity=8)
        dets = []
        for k in range(1, n):
            comp = labels == k
            peak = float(up[comp].max())
            if peak >= self.presence_threshold:
                dets.append(Detection(bbox(comp), peak))
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets[:self.max_candidates]


# ── Tracker (M2–M4) ────────────────────────────────────────────────────────

class PatchMemory:
    def __init__(self, recent: int, max_fg: int, max_bg: int, seed: int = 0):
        self.max_fg, self.max_bg = max_fg, max_bg
        self.anchor = None
        self.recent = deque(maxlen=recent)
        self.rng = np.random.default_rng(seed)

    def _cap(self, x, n):
        return x if len(x) <= n else x[self.rng.permutation(len(x))[:n]]

    def set_anchor(self, fg, bg):
        self.anchor = (self._cap(fg, self.max_fg), self._cap(bg, self.max_bg))
        self.recent.clear()

    def add(self, fg, bg):
        self.recent.append((self._cap(fg, self.max_fg // 2), self._cap(bg, self.max_bg // 2)))

    def banks(self):
        slots = [self.anchor] + list(self.recent)
        return np.concatenate([s[0] for s in slots]), np.concatenate([s[1] for s in slots])


class MobileTracker:
    def __init__(self, model_dir: str = "mobile/models", providers: Optional[List[str]] = None,
                 int8: bool = False, cfg: Optional[dict] = None):
        self.models = Models(model_dir, providers, int8)
        self.grounder = Grounder(self.models)
        self.cfg = {**self.models.manifest["tracker_cfg"], **(cfg or {})}
        self.reset()

    def reset(self):
        c = self.cfg
        self.prompt = None
        self.memory = PatchMemory(c["recent_keyframes"], c["max_fg_bank"], c["max_bg_bank"])
        self.frame_t = 0
        self.box = None
        self.area_ema = None
        self.low_conf_count = 0
        self.last_reground_t = 0
        self.absent = False
        self.pending_box = None
        self.prev_thumb = None
        self.prev_feats = None
        self.gray_diffs = deque(maxlen=30)
        self.last_conf = 0.0
        self.last_event = None

    @staticmethod
    def _as_float(frame: np.ndarray) -> np.ndarray:
        return frame.astype(np.float32) / 255.0 if frame.dtype == np.uint8 else frame.astype(np.float32)

    # Public API -------------------------------------------------------------

    def init(self, frame: np.ndarray, prompt: str) -> Optional[np.ndarray]:
        """frame: HxWx3 RGB (uint8, or float in [0, 1])."""
        frame = self._as_float(frame)
        self.reset()
        self.prompt = prompt
        ff = self.models.dino_patches(frame)
        self._scene_cut(frame, ff)
        dets = self.grounder.detect(frame, prompt)
        if not dets:
            self.absent, self.last_event = True, "absent"
            return None
        mask = self._detection_mask(frame, dets[0])
        self._commit(ff, mask, anchor=True)
        self.last_event = "init"
        return mask

    def step(self, frame: np.ndarray) -> Optional[np.ndarray]:
        frame = self._as_float(frame)
        c = self.cfg
        self.frame_t += 1
        self.last_event = None
        ff = self.models.dino_patches(frame)
        cut = self._scene_cut(frame, ff)

        if self.memory.anchor is None or self.absent:
            if cut or self.frame_t - self.last_reground_t >= c["absent_retry"]:
                return self._reground(frame, ff, "scene_cut" if cut else "absent")
            return None
        if cut:
            return self._reground(frame, ff, "scene_cut")

        mask, conf = self._propagate(frame, ff)
        self.last_conf = conf
        self.low_conf_count = self.low_conf_count + 1 if (mask is None or conf < c["low_conf"]) else 0
        cooled = self.frame_t - self.last_reground_t >= c["reground_cooldown"]
        if self.low_conf_count >= c["low_conf_frames"] and cooled:
            return self._reground(frame, ff, "low_confidence", fallback=mask)
        if c["reground_interval"] and self.frame_t - self.last_reground_t >= c["reground_interval"]:
            return self._reground(frame, ff, "periodic", fallback=mask)
        if mask is not None:
            area_ok = self.area_ema is None or 0.5 <= mask.sum() / self.area_ema <= 2.0
            write = self.frame_t % c["keyframe_interval"] == 0 and conf >= c["keyframe_min_conf"] and area_ok
            self._commit(ff, mask, keyframe=write)
        return mask

    # Propagation ------------------------------------------------------------

    def _scores(self, feats):
        fg, bg = self.memory.banks()
        k = self.cfg["knn"]
        return topk_mean(feats @ fg.T, k) - topk_mean(feats @ bg.T, k)

    def _propagate(self, frame, ff):
        c = self.cfg
        H, W = frame.shape[:2]
        feats, gh, gw = ff
        score = self._scores(feats).reshape(gh, gw)
        x1, y1, x2, y2 = self.box
        m = c["search_margin"] * max(x2 - x1, y2 - y1)
        window = np.zeros((gh, gw), bool)
        window[max(0, int((y1 - m) / H * gh)):min(gh, int(np.ceil((y2 + m) / H * gh))),
               max(0, int((x1 - m) / W * gw)):min(gw, int(np.ceil((x2 + m) / W * gw)))] = True
        n, labels = cv2.connectedComponents(((score > 0) & window).astype(np.uint8), connectivity=8)
        if n <= 1:
            return None, float(score[window].max())

        prev = np.zeros((gh, gw), bool)
        prev[int(y1 / H * gh):int(np.ceil(y2 / H * gh)), int(x1 / W * gw):int(np.ceil(x2 / W * gw))] = True
        best = max(range(1, n), key=lambda k: ((labels == k)[prev].sum(), score[labels == k].clip(min=0).sum()))
        comp = labels == best
        conf = float(score[comp].mean())

        ys, xs = np.where(comp)
        cw, ch = W / gw, H / gh
        coarse = np.array([xs.min() * cw, ys.min() * ch, (xs.max() + 1) * cw, (ys.max() + 1) * ch], np.float32)
        gate = resize(cv2.dilate(comp.astype(np.float32), np.ones((3, 3), np.uint8)), H, W) > 0.5

        mask = (self._mask_probs(frame, coarse) > c["mask_threshold"]) & gate
        if mask.sum() < 0.15 * gate.sum():
            mask = (resize(score.astype(np.float32), H, W) > 0) & gate
        if not mask.any():
            return None, conf
        return mask, conf

    # Re-grounding -----------------------------------------------------------

    def _reground(self, frame, ff, reason, fallback=None):
        self.last_reground_t = self.frame_t
        det = self._select(frame, ff)
        if det is None:
            if fallback is not None and reason == "periodic":
                self._commit(ff, fallback)
                return fallback
            self.absent, self.low_conf_count, self.pending_box = True, 0, None
            self.last_event = "absent"
            return None
        if self.absent or self.memory.anchor is None:          # confirm re-acquisition
            pending, self.pending_box = self.pending_box, det.box
            if pending is None or box_iou(pending, det.box) < 0.3:
                self.absent, self.last_event = True, "absent"
                return None
        self.pending_box = None
        mask = self._detection_mask(frame, det)
        first = self.memory.anchor is None
        self._commit(ff, mask, anchor=first, keyframe=not first)
        self.absent, self.low_conf_count = False, 0
        self.last_event = f"reground:{reason}"
        return mask

    def _select(self, frame, ff) -> Optional[Detection]:
        dets = self.grounder.detect(frame, self.prompt)
        if not dets:
            return None
        if self.memory.anchor is None:
            return dets[0]
        feats, gh, gw = ff
        score = self._scores(feats).reshape(gh, gw)
        best, best_val = None, -1e9
        for d in dets:
            d.mask = self._detection_mask(frame, d)
            grid = grid_fraction(d.mask, gh, gw) > 0.5
            appearance = float(score[grid].mean()) if grid.any() else -1.0
            if appearance < self.cfg["min_det_appearance"] and d.score < 0.6:
                continue
            if d.score + 4.0 * appearance > best_val:
                best, best_val = d, d.score + 4.0 * appearance
        return best

    def _detection_mask(self, frame, det: Detection) -> np.ndarray:
        if det.mask is not None and det.mask.any():
            return det.mask
        H, W = frame.shape[:2]
        box, mask = det.box, None
        for _ in range(self.cfg["box_refine_iters"]):
            m = (self._mask_probs(frame, box) > self.cfg["mask_threshold"]) & box_mask(box, H, W, pad=0.15)
            if not m.any():
                break
            mask, box = m, bbox(m)
        return mask if mask is not None else box_mask(det.box, H, W)

    # State ------------------------------------------------------------------

    def _commit(self, ff, mask, anchor=False, keyframe=False):
        if not mask.any():
            return
        feats, gh, gw = ff
        if anchor or keyframe:
            frac = grid_fraction(mask, gh, gw).ravel()
            fg = feats[frac > 0.5]
            if len(fg) == 0:
                fg = feats[frac.argmax()][None]
            bg = feats[frac < 0.05]
            (self.memory.set_anchor if anchor else self.memory.add)(fg, bg)
        self.box = bbox(mask)
        area = float(mask.sum())
        self.area_ema = area if self.area_ema is None else 0.8 * self.area_ema + 0.2 * area

    def _mask_probs(self, frame, box) -> np.ndarray:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = box
        pad = max(x2 - x1, y2 - y1, 1.0) * 0.15
        ex1, ey1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
        ex2, ey2 = min(W, int(np.ceil(x2 + pad))), min(H, int(np.ceil(y2 + pad)))
        full = np.zeros((H, W), np.float32)
        if ex2 - ex1 < 2 or ey2 - ey1 < 2:
            return full
        probs = self.models.segment(frame[ey1:ey2, ex1:ex2])
        full[ey1:ey2, ex1:ex2] = resize(probs, ey2 - ey1, ex2 - ex1)
        return full

    def _scene_cut(self, frame, ff) -> bool:
        c = self.cfg
        thumb = resize(frame.mean(2), 36, 64, area=True) * 255
        feats = ff[0]
        cut = False
        if self.prev_thumb is not None:
            diff = float(np.abs(thumb - self.prev_thumb).mean())
            median = float(np.median(self.gray_diffs)) if self.gray_diffs else 0.0
            cut = diff > c["cut_gray_diff"] and diff > c["cut_gray_ratio"] * median
            if self.prev_feats is not None and self.prev_feats.shape == feats.shape:
                cut = cut or float(((feats * self.prev_feats).sum(1) < 0.5).mean()) > c["cut_patch_frac"]
            if not cut:
                self.gray_diffs.append(diff)
        self.prev_thumb, self.prev_feats = thumb, feats
        return cut
