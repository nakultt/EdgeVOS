# EdgeVOS

Text-prompted video object segmentation for edge devices.

```
M1  text grounding      MobileCLIP2-S2 dense heatmap (one forward pass)
M2  mask decoding       RepViT-M1 backbone + SpatialMaskHead (distilled from SAM 3.1)
M3  propagation         DINOv3 ViT-S/16 patch features + gated fg/bg memory
M4  re-grounding        scene-cut / low-confidence triggered call back into M1
```

SAM 3 is the **teacher only**: it produced the distillation targets and serves as
an offline reference (`EdgeVOSPredictor(..., use_sam3=True)`). It is not part of
the product path.

## Setup

Python 3.11 and an NVIDIA GPU with CUDA 12.x are expected.

```bash
git clone --recurse-submodules https://github.com/nakultt/EdgeVOS.git
```

The `sam3` submodule is only needed for distillation and offline evaluation:

```bash
git submodule update --init sam3
```

Create the environment (Windows paths shown; use `.venv/bin/python` on Linux):

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

Model weights: `checkpoints/edgevos_spatial_v1.pt` (bundled by `scripts/bundle_checkpoint.py`).
DINOv3 and MobileCLIP2-S2 weights download from Hugging Face on first use.
For the SAM 3 teacher (dev only), `facebook/sam3` is gated; run `huggingface-cli login` once after accepting its license.

## Run

```bash
.venv/Scripts/python a.py
```

Tracks `"cat"` in `b.mp4` and writes `tracked.mp4`. For any video and prompt:

```bash
.venv/Scripts/python demo_mp4.py --video_path b.mp4 --prompt "cat" --output tracked.mp4
```

In the output, green frames were (re-)grounded from the text prompt, red frames were tracked, and frames without an overlay are frames where the object is absent.

## Python API

```python
from edgevos import EdgeVOSPredictor

model = EdgeVOSPredictor("checkpoints/edgevos_spatial_v1.pt")        # SAM-free product path
mask = model.init_from_text(model.frame_to_tensor(frame_bgr), "cat")  # [H, W] bool, or None
mask = model.process_frame(model.frame_to_tensor(next_frame_bgr))    # None while the object is absent
model.last_event                                                      # 'init' | 'reground:<reason>' | 'absent' | None
```

`frame_to_tensor` accepts OpenCV BGR uint8 arrays (or RGB with `bgr=False`) and tensors. Tracker settings are in `inference/tracker.py` (`DEFAULT_TRACKER_CFG`) and can be overridden with `EdgeVOSPredictor(..., tracker_cfg={...})`.

## On device

The SAM-free pipeline exports to ONNX / ONNX Runtime Mobile and runs without PyTorch.
See [mobile/README.md](mobile/README.md):

```bash
.venv/Scripts/python scripts/export_mobile.py --int8 --ort
```

```bash
.venv/Scripts/python -m mobile.demo --video b.mp4 --prompt cat --int8
```
