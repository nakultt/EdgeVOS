# EdgeVOS

Text-prompted video object segmentation for edge devices.

```
M1  text grounding      SAM 3.1 (fallback: MobileCLIP2-S2 dense window scoring)
M2  mask decoding       RepViT-M1 backbone + SpatialMaskHead (distilled from SAM 3.1)
M3  propagation         DINOv3 ViT-S/16 patch features + STGP graph memory
M4  re-grounding        confidence / scene-cut triggered call back into M1
```

## Setup

Python 3.11 and an NVIDIA GPU with CUDA 12.x are expected.

```bash
git clone --recurse-submodules https://github.com/nakultt/EdgeVOS.git
```

If the repo is already cloned, fetch the `sam3` submodule:

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
DINOv3, MobileCLIP2-S2 and SAM 3 weights download from Hugging Face on first use.
`facebook/sam3` is a gated model; run `huggingface-cli login` once after accepting its license.

## Run

```bash
.venv/Scripts/python demo_mp4.py --video_path b.mp4 --prompt "cat" --output tracked.mp4
```
