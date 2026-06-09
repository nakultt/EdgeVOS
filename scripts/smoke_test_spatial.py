"""Smoke test: verify SpatialMaskHead and new StudentModel work correctly."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.backbone import BackboneWithRoI
from models.mask_head import SpatialMaskHead, MLPMaskHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Test SpatialMaskHead standalone ─────────────────────────
print("\n1) SpatialMaskHead standalone test...")
head = SpatialMaskHead(in_channels=384, out_size=256).to(device)
feat_map = torch.randn(2, 384, 14, 14, device=device)
out = head(feat_map)
print(f"   Input:  {feat_map.shape}")
print(f"   Output: {out.shape}")
assert out.shape == (2, 1, 256, 256), f"Expected (2,1,256,256), got {out.shape}"
print("   ✓ Shape correct!")

# Verify gradient flow
loss = out.sum()
loss.backward()
has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
               for p in head.parameters() if p.requires_grad)
assert has_grad, "No gradients!"
print("   ✓ Gradients flow!")

# ── Test full StudentModel path ─────────────────────────────
print("\n2) Full StudentModel path test...")
backbone = BackboneWithRoI(pretrained=True).to(device)
head2 = SpatialMaskHead(in_channels=backbone.feat_channels, out_size=256).to(device)

images = torch.randn(2, 3, 224, 224, device=device)
with torch.no_grad():
    feat_map = backbone._extract_feature_map(images)  # [B, 384, 14, 14]
    mask_logits = head2(feat_map)                      # [B, 1, 256, 256]

print(f"   Backbone feature map: {feat_map.shape}")
print(f"   Mask logits:          {mask_logits.shape}")
assert feat_map.shape == (2, 384, 14, 14)
assert mask_logits.shape == (2, 1, 256, 256)
print("   ✓ End-to-end correct!")

# ── Parameter count comparison ──────────────────────────────
print("\n3) Parameter counts...")
mlp_head = MLPMaskHead(in_dim=384, hidden_dims=(512, 256), out_size=256)
spatial_head = SpatialMaskHead(in_channels=384, out_size=256)

mlp_params = sum(p.numel() for p in mlp_head.parameters())
spatial_params = sum(p.numel() for p in spatial_head.parameters())

print(f"   MLPMaskHead:     {mlp_params:>10,} params ({mlp_params * 4 / 1024 / 1024:.1f} MB)")
print(f"   SpatialMaskHead: {spatial_params:>10,} params ({spatial_params * 4 / 1024 / 1024:.1f} MB)")

print("\n✓ All smoke tests passed! Ready to retrain.")
