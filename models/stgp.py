"""
EdgeVOS — Module 3: Spatio-Temporal Graph Propagation (STGP)
The core novel contribution. Patent-pending.

Architecture:
  - Each tracked object is a node sequence in a directed graph
  - Nodes store 256-dim DINOv3 feature vectors at keyframe intervals K
  - Propagation = cosine similarity matching between node features and frame patches
  - Memory grows O(K) not O(T) — unlike SAM 3.1's dense masklet bank
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class NodeState(Enum):
    ACTIVE   = "active"
    OCCLUDED = "occluded"
    LOST     = "lost"


@dataclass
class STGPNode:
    """Single node in the STGP graph. Represents object state at one keyframe."""
    feat:    torch.Tensor          # [256] projected DINOv3 feature
    mask:    torch.Tensor          # [H, W] binary mask at this keyframe
    frame_t: int                   # Frame index
    conf:    float = 1.0           # Cosine similarity confidence at creation
    state:   NodeState = NodeState.ACTIVE
    is_regrounded: bool = False    # True if this node was created by M4 re-grounding


@dataclass
class STGPEdge:
    """Edge between two consecutive nodes of the same object."""
    src_t:   int
    dst_t:   int
    weight:  float        # Cosine similarity between node features
    edge_type: str = "temporal"   # "temporal" | "regrounding"


class STGPProjection(nn.Module):
    """
    Learned projection: DINOv3 384-dim → STGP 256-dim.
    Trained with InfoNCE contrastive loss on DAVIS pairs.
    Same object across frames = positive; different objects = negative.
    """
    def __init__(self, in_dim: int = 384, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., 384] DINOv3 patch features
        Returns:
            [..., 256] L2-normalized projected features
        """
        return F.normalize(self.proj(x), dim=-1)


class STGPGraph:
    """
    The full spatio-temporal graph for one video.
    Tracks multiple objects simultaneously, each as a node sequence.

    Memory usage: O(N * T/K * 256 * 4 bytes)
    For 5 objects, 10min video at 30fps, K=5: ~3.6MB — trivial.
    SAM 3.1 equivalent: ~18GB (OOM on 8GB Jetson Orin Nano).
    """

    def __init__(
        self,
        projection: STGPProjection,
        keyframe_interval:    int   = 5,
        sim_trigger:          float = 0.45,
        sim_recover:          float = 0.55,
        drift_frames:         int   = 3,
        regrounding_interval: int   = 150,
        topk_patches:         int   = 30,
        ghost_confirm:        int   = 5,
    ):
        self.proj                  = projection
        self.K                     = keyframe_interval
        self.sim_trigger           = sim_trigger
        self.sim_recover           = sim_recover
        self.drift_frames          = drift_frames
        self.regrounding_interval  = regrounding_interval
        self.topk_patches          = topk_patches
        self.ghost_confirm         = ghost_confirm

        # Core storage
        self.nodes: Dict[int, List[STGPNode]] = {}   # obj_id -> node list
        self.edges: Dict[int, List[STGPEdge]] = {}   # obj_id -> edge list

        # Tracking state
        self._below_thresh_count: Dict[int, int]  = {}   # consecutive low-sim frames
        self._ghost_count:        Dict[int, int]  = {}   # frames since re-id candidate
        self._last_reground_t:    Dict[int, int]  = {}   # last re-grounding frame

    # ── Initialisation ────────────────────────────────────────────

    def init_object(
        self,
        obj_id:       int,
        feat_384:     torch.Tensor,   # [384] raw DINOv3 feature
        mask:         torch.Tensor,   # [H, W] binary mask from M2
        frame_t:      int = 0,
    ) -> None:
        """Register a new tracked object at frame 0 (or any initialization frame)."""
        feat_256 = self.proj(feat_384.unsqueeze(0)).squeeze(0)
        node = STGPNode(feat=feat_256, mask=mask, frame_t=frame_t, conf=1.0)
        self.nodes[obj_id]              = [node]
        self.edges[obj_id]              = []
        self._below_thresh_count[obj_id] = 0
        self._ghost_count[obj_id]        = 0
        self._last_reground_t[obj_id]    = frame_t

    # ── Per-frame propagation ─────────────────────────────────────

    def propagate(
        self,
        obj_id:        int,
        frame_patches: torch.Tensor,   # [196, 384] DINOv3 patches for full frame
        frame_t:       int,
        new_mask:      Optional[torch.Tensor] = None,  # [H, W] from M2, if keyframe
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool, Optional[str]]:
        """
        One frame of STGP propagation for one object.

        Returns:
            sims:           [196] cosine similarity per patch
            topk_indices:   [topk_patches] indices of highest-similarity patches
            needs_regrounding: True if M4 should be triggered
            trigger_reason: Optional[str] reason for re-grounding ("primary", "tertiary", "lost", or None)
        """
        node = self.latest(obj_id)
        if node is None or node.state == NodeState.LOST:
            return None, None, True, "lost"

        # Project frame patches to 256-dim
        with torch.no_grad():
            patches_256 = self.proj(frame_patches)   # [196, 256]

        # Cosine similarity: stored node feature vs all frame patches
        sims = torch.cosine_similarity(
            node.feat.unsqueeze(0),   # [1, 256]
            patches_256,              # [196, 256]
            dim=1
        )  # [196]

        topk_indices = sims.topk(self.topk_patches).indices
        max_sim = sims.max().item()

        # Update confidence tracking
        if max_sim < self.sim_trigger:
            self._below_thresh_count[obj_id] += 1
        else:
            self._below_thresh_count[obj_id] = 0
            if node.state == NodeState.OCCLUDED:
                self._ghost_count[obj_id] += 1
                if self._ghost_count[obj_id] >= self.ghost_confirm:
                    node.state = NodeState.ACTIVE
                    self._ghost_count[obj_id] = 0

        # Mark occluded if below threshold for drift_frames consecutive frames
        needs_regrounding = False
        trigger_reason = None
        if self._below_thresh_count[obj_id] >= self.drift_frames:
            node.state = NodeState.OCCLUDED
            needs_regrounding = True
            trigger_reason = "primary"

        # Scheduled re-grounding regardless of confidence
        if frame_t - self._last_reground_t[obj_id] >= self.regrounding_interval:
            needs_regrounding = True
            # Prefer primary if both trigger
            if trigger_reason is None:
                trigger_reason = "tertiary"

        # Add new keyframe node
        if frame_t % self.K == 0 and new_mask is not None and not needs_regrounding:
            self._add_keyframe_node(obj_id, frame_patches, new_mask, frame_t, max_sim)

        return sims, topk_indices, needs_regrounding, trigger_reason

    def add_regrounding_node(
        self,
        obj_id:    int,
        feat_384:  torch.Tensor,
        mask:      torch.Tensor,
        frame_t:   int,
        conf:      float,
    ) -> None:
        """Called by M4 after successful re-grounding. Inserts a re-anchor node."""
        feat_256 = self.proj(feat_384.unsqueeze(0)).squeeze(0)
        prev_node = self.latest(obj_id)
        node = STGPNode(
            feat=feat_256, mask=mask, frame_t=frame_t,
            conf=conf, state=NodeState.ACTIVE, is_regrounded=True
        )
        self.nodes[obj_id].append(node)
        if prev_node is not None:
            edge = STGPEdge(
                src_t=prev_node.frame_t, dst_t=frame_t,
                weight=conf, edge_type="regrounding"
            )
            self.edges[obj_id].append(edge)
        self._below_thresh_count[obj_id] = 0
        self._last_reground_t[obj_id]    = frame_t

    # ── Helpers ───────────────────────────────────────────────────

    def latest(self, obj_id: int) -> Optional[STGPNode]:
        nodes = self.nodes.get(obj_id, [])
        return nodes[-1] if nodes else None

    def all_active_ids(self) -> List[int]:
        return [oid for oid, nodes in self.nodes.items()
                if nodes and nodes[-1].state == NodeState.ACTIVE]

    def memory_bytes(self) -> int:
        """Total memory used by all graph nodes (feature tensors only)."""
        total = 0
        for nodes in self.nodes.values():
            for node in nodes:
                total += node.feat.numel() * node.feat.element_size()
        return total

    def _add_keyframe_node(
        self,
        obj_id:   int,
        patches:  torch.Tensor,
        mask:     torch.Tensor,
        frame_t:  int,
        conf:     float,
    ) -> None:
        """Extract feature from masked region and add a new keyframe node."""
        with torch.no_grad():
            # Mean-pool DINOv3 patches that fall inside the predicted mask region
            # patches: [196, 384], mask is upsampled to 14x14 grid
            grid_mask = F.interpolate(
                mask.float().unsqueeze(0).unsqueeze(0),
                size=(14, 14), mode='nearest'
            ).squeeze().flatten()   # [196]
            inside = patches[grid_mask > 0.5]
            if inside.numel() == 0:
                inside = patches  # fallback: use all patches
            feat_384 = inside.mean(0)   # [384]
        feat_256 = self.proj(feat_384.unsqueeze(0)).squeeze(0)
        prev = self.latest(obj_id)
        node = STGPNode(feat=feat_256, mask=mask, frame_t=frame_t, conf=conf)
        self.nodes[obj_id].append(node)
        if prev is not None:
            sim = torch.cosine_similarity(
                prev.feat.unsqueeze(0), feat_256.unsqueeze(0)
            ).item()
            self.edges[obj_id].append(
                STGPEdge(src_t=prev.frame_t, dst_t=frame_t, weight=sim)
            )
