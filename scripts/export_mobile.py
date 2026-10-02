"""
Export the SAM-free EdgeVOS product models for on-device inference.

Writes to mobile/models/:
  dinov3_patches.onnx   image  [1,3,H,W] RGB 0..1 (H, W multiples of 16) -> patches [H/16*W/16, 384], L2-normalised
  segmenter.onnx        crop   [1,3,224,224] RGB 0..1                    -> probs   [1,1,256,256]
  clip_dense.onnx       image  [1,3,H,W] RGB 0..1 (H, W multiples of 32) -> dense   [H/32, W/32, 512], L2-normalised
  clip_text.onnx        tokens [1,77] int64                              -> text    [1, 512], L2-normalised
  clip_background.npy   background prompt embeddings used by the grounder
  manifest.json         files, shapes, sizes, tracker config, grounding settings, parity results
plus INT8 weight-quantised copies of the transformer models (--int8) and
ORT-format files for ONNX Runtime Mobile (--ort).

ImageNet / CLIP normalisation, register-token slicing, L2-normalisation and the
sigmoid are baked into the graphs: the device only feeds RGB float images in
[0, 1]. H and W are dynamic; on NNAPI / CoreML fix them per session with ONNX
Runtime free-dimension overrides ("H", "W").

Usage:
  python scripts/export_mobile.py --int8 --ort
"""

import argparse
import inspect
import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.backbone import BackboneWithRoI          # noqa: E402
from models.mask_head import SpatialMaskHead         # noqa: E402
from models.grounding import CLIPGrounder            # noqa: E402
from inference.tracker import DEFAULT_TRACKER_CFG    # noqa: E402
from edgevos.predictor import CLIP_TRACKER_CFG       # noqa: E402

# The on-device tracker runs exactly the SAM-free desktop configuration.
TRACKER_CFG = {**DEFAULT_TRACKER_CFG, **CLIP_TRACKER_CFG}
GROUNDER_LONG_SIDE = inspect.signature(CLIPGrounder.__init__).parameters["long_side"].default


class DinoPatches(nn.Module):
    def __init__(self, dino):
        super().__init__()
        self.dino = dino
        self.n_prefix = 1 + int(getattr(dino.config, "num_register_tokens", 0))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        tokens = self.dino(pixel_values=(image - self.mean) / self.std).last_hidden_state
        return F.normalize(tokens[0, self.n_prefix:], dim=-1)


class Segmenter(nn.Module):
    def __init__(self, backbone, mask_head):
        super().__init__()
        self.backbone, self.mask_head = backbone, mask_head

    def forward(self, crop):
        return self.mask_head(self.backbone._extract_feature_map(crop)).sigmoid()


class ClipDense(nn.Module):
    def __init__(self, trunk, mean, std):
        super().__init__()
        self.trunk = trunk
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, image):
        fmap = self.trunk.forward_features((image - self.mean) / self.std)
        return F.normalize(self.trunk.head.fc(fmap.permute(0, 2, 3, 1))[0], dim=-1)


class ClipText(nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, tokens):
        return F.normalize(self.clip.encode_text(tokens), dim=-1)


def natural_image(h: int, w: int, seed: int) -> torch.Tensor:
    """Smooth, image-like test input. Uniform noise drives RepViT to extreme
    activations (~1e5) and makes fp32 parity checks meaningless."""
    g = torch.Generator().manual_seed(seed)
    low = torch.rand(1, 3, max(2, h // 32), max(2, w // 32), generator=g)
    img = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)
    return (img + 0.03 * torch.randn(1, 3, h, w, generator=g)).clamp(0, 1)


def export(model, example, path, input_name, output_name, dynamic_hw=False, checks=()):
    """Export, then compare ONNX Runtime against PyTorch on ``checks`` inputs."""
    import onnxruntime as ort

    model = model.eval().requires_grad_(False)
    torch.backends.mha.set_fastpath_enabled(False)   # fused MHA kernel has no ONNX symbolic
    dyn = {input_name: {2: "H", 3: "W"}} if dynamic_hw else None
    torch.onnx.export(model, (example,), path, input_names=[input_name], output_names=[output_name],
                      dynamic_axes=dyn, opset_version=17, dynamo=False)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    rel_err = 0.0
    for x in checks or (example,):
        with torch.no_grad():
            ref = model(x).numpy()
        out = sess.run(None, {input_name: x.numpy()})[0]
        rel_err = max(rel_err, float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-9)))
    shape = list(example.shape)
    if dynamic_hw:
        shape[2:] = ["H", "W"]
    return {"file": os.path.basename(path), "input": {"name": input_name, "shape": shape, "dtype": str(example.dtype)},
            "output": output_name, "max_rel_err": rel_err, "mb": round(os.path.getsize(path) / 2**20, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/edgevos_spatial_v1.pt")
    ap.add_argument("--out", default="mobile/models")
    ap.add_argument("--int8", action="store_true", help="also write INT8 weight-quantised transformer models")
    ap.add_argument("--ort", action="store_true", help="also convert to ORT format for ONNX Runtime Mobile")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModel
    import open_clip

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, sd = ckpt.get("config", {}), ckpt["state_dict"]
    manifest = {"input_format": "RGB float32 in [0,1], NCHW", "tracker_cfg": TRACKER_CFG, "models": {}}
    path = lambda name: os.path.join(args.out, name)

    print("DINOv3 patches ...")
    dino = AutoModel.from_pretrained(cfg.get("dinov3_model", "facebook/dinov3-vits16-pretrain-lvd1689m")).eval()
    info = export(DinoPatches(dino), natural_image(256, 448, 0), path("dinov3_patches.onnx"), "image", "patches",
                  dynamic_hw=True, checks=[natural_image(256, 448, 1), natural_image(448, 256, 2)])
    manifest["models"]["dinov3_patches"] = {**info, "stride": 16, "long_side": TRACKER_CFG["dino_long_side"]}

    print("Segmenter (RepViT + SpatialMaskHead) ...")
    bb = BackboneWithRoI(model_name=cfg.get("backbone", "repvit_m1"), pretrained=False)
    mh = SpatialMaskHead(bb.feat_channels, cfg.get("mask_out_size", 256))
    bb.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")})
    mh.load_state_dict({k[len("mask_head."):]: v for k, v in sd.items() if k.startswith("mask_head.")})
    manifest["models"]["segmenter"] = export(Segmenter(bb, mh), natural_image(224, 224, 0), path("segmenter.onnx"),
                                             "crop", "probs", checks=[natural_image(224, 224, s) for s in (1, 2, 3)])

    print("MobileCLIP2-S2 ...")
    clip, _, preprocess = open_clip.create_model_and_transforms("MobileCLIP2-S2", pretrained="dfndr2b")
    clip.eval()
    grounder = CLIPGrounder(clip, preprocess, open_clip.get_tokenizer("MobileCLIP2-S2"), device="cpu")
    info = export(ClipDense(clip.visual.trunk, grounder.mean, grounder.std), natural_image(576, 1024, 0),
                  path("clip_dense.onnx"), "image", "dense", dynamic_hw=True,
                  checks=[natural_image(576, 1024, 1), natural_image(1024, 576, 2)])
    manifest["models"]["clip_dense"] = {**info, "stride": 32, "long_side": GROUNDER_LONG_SIDE}
    tokens = open_clip.get_tokenizer("MobileCLIP2-S2")(["a photo of a cat."])
    manifest["models"]["clip_text"] = export(ClipText(clip), tokens, path("clip_text.onnx"), "tokens", "text")
    np.save(path("clip_background.npy"), grounder._bg_feat.numpy().astype(np.float32))
    manifest["grounding"] = {
        "prompt_template": "a photo of a {}.",
        "background_prompts": [f"a photo of {p}." for p in CLIPGrounder.BACKGROUND_PROMPTS],
        "presence_threshold": grounder.presence_threshold,
        "tokenizer": "CLIP BPE, context 77: mobile/tokenizer.py + mobile/assets/bpe_simple_vocab_16e6.txt.gz",
    }

    if args.int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        for name in ("dinov3_patches", "clip_text"):
            src = path(manifest["models"][name]["file"])
            dst = src.replace(".onnx", ".int8.onnx")
            quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
            manifest["models"][name]["int8_file"] = os.path.basename(dst)
            manifest["models"][name]["int8_mb"] = round(os.path.getsize(dst) / 2**20, 1)

    if args.ort:
        subprocess.run([sys.executable, "-m", "onnxruntime.tools.convert_onnx_models_to_ort", args.out,
                        "--optimization_style", "Fixed"], check=True)
        manifest["ort_format"] = True

    with open(path("manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    for name, m in manifest["models"].items():
        extra = f"   int8 {m['int8_mb']:6.1f} MB" if "int8_mb" in m else ""
        print(f"  {name:16s} {m['mb']:7.1f} MB   max rel err {m['max_rel_err']:.1e}{extra}")
    print(f"Wrote {path('manifest.json')}")


if __name__ == "__main__":
    main()
