# EdgeVOS

Text-prompted video object segmentation that runs on a phone. Type what to follow ("cat", "bottle",
"keyboard"), and EdgeVOS finds it, segments it and tracks it frame by frame. It re-acquires the object
after scene cuts and reports when it's gone.

SAM 3 was used as the **teacher only**. The product path is small distilled and off-the-shelf models
that run fully on-device, on the Hexagon NPU of a Snapdragon phone, at **27–30 fps**.

**Download the Android app:** [Releases → `android-v1.0`](https://github.com/nakultt/EdgeVOS/releases/tag/android-v1.0)
(arm64, Android 10+). The NPU path needs a Snapdragon 8 Gen 3 (Hexagon V75), tested on a Galaxy S24. Other
phones fall back to the CPU at a few fps (fallback not tested on non-Snapdragon phones).

## How it works

```
M1  text grounding   MobileCLIP2-S2 dense heatmap: prompt vs background, one forward pass
M2  mask decoding    RepViT-M1 + SpatialMaskHead, distilled from SAM 3.1 (box crop → mask)
M3  propagation      DINOv3 ViT-S/16 patch features + kNN fg/bg patch memory (anchor + keyframes)
M4  re-grounding     on scene cuts, sustained low confidence, absence, and periodically → back to M1
```

1. **Start:** the prompt is encoded once ("a photo of a {prompt}."). CLIP's dense heatmap against a set of
   background prompts gives candidate regions, and the mask model refines the best candidate into a mask
   (green in the demos and the app).
2. **Every frame:** DINOv3 patch features are classified against the foreground/background memory inside a
   motion window around the last box. The mask model turns that region into the mask (red).
3. **Re-grounding:** a scene cut, a run of low confidence, or a periodic check calls M1 again. Losing the
   object reports "absent" (no mask), and re-acquiring it has to be confirmed on two retries.

## Results

| | |
|---|---|
| **Quality** (b.mp4 "cat", mean mask IoU vs SAM 3, Python) | **0.864** SAM-free (0.888 with SAM 3 doing the grounding) |
| **Absence** (b.mp4 "car", not in the video) | correctly absent on every frame |
| **Phone speed** (Galaxy S24, Hexagon NPU, release APK) | **27–30 fps**, 37 ms per tracked frame (DINOv3 11, scoring 9, segmenter 3, YUV 4); scene-cut re-grounding ~150 ms |
| **Startup** | 89 s once after install (NPU compile), **1.8 s** afterwards |
| **Per-model NPU time** | DINOv3 9.2 ms, segmenter 1.5 ms, CLIP 50 ms (at re-grounds only); 10–30× faster than the phone CPU |
| **Port fidelity** | Kotlin vs Python on the CPU: mean IoU 1.0000, identical events; phone NPU (fp16) vs Python: 0.9554 |

Details: [`mobile/README.md`](mobile/README.md) (models, phone benchmark) and
[`android/README.md`](android/README.md) (app, measurements, parity).

**Known limits:** grounding is CLIP-based, so it works best with everyday object nouns. Body parts
("hand") and prompts close to the background prompts ("plant", "furniture", "wall") ground poorly. Two
adjacent objects can merge into one mask (in b.mp4 the cat and the toy dog next to it). Tracked-frame edges
follow the 16-px patch grid in places.

## Repository layout

| Path | What |
|---|---|
| [`edgevos/`](edgevos) | `EdgeVOSPredictor`, the Python API (PyTorch) |
| [`inference/`](inference) | streaming tracker (`tracker.py`, M3/M4) and the pipeline |
| [`models/`](models) | backbone, `SpatialMaskHead`, CLIP grounder, STGP, re-grounding |
| [`train/`](train) | distillation of the mask head from SAM 3.1 (`distill_mask.py`), STGP training |
| [`scripts/`](scripts) | SAM 3.1 target caching, checkpoint bundling, ONNX export (`export_mobile.py`), DAVIS runs, debugging tools |
| [`eval/`](eval) | DAVIS-2017 J&F evaluation and results |
| [`configs/`](configs) | Hydra training config |
| [`data/`](data) | DAVIS dataset loader |
| [`mobile/`](mobile) | PyTorch-free reference runtime (ONNX Runtime + NumPy), tokenizer, phone benchmark |
| [`android/`](android) | Android app (`app/`), Kotlin port of the runtime (`core/`), parity tools (`tools/`) |
| [`demos/`](demos) | runnable examples |
| [`samples/`](samples) | sample clips (`b.mp4` is the reference clip for all quality and parity numbers) |
| `sam3/` | SAM 3 submodule, teacher and offline reference only |

## Setup (Python)

Python 3.11 and an NVIDIA GPU with CUDA 12.x are expected for the PyTorch path.

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

From the repo root:

```bash
.venv/Scripts/python -m demos.quickstart
```

Tracks `"cat"` in `samples/b.mp4` and writes `tracked.mp4`. For any video and prompt:

```bash
.venv/Scripts/python -m demos.demo_mp4 --video_path samples/b.mp4 --prompt "cat" --output tracked.mp4
```

Green frames were (re-)grounded from the text prompt, red frames were tracked, and frames without an overlay
are frames where the object is absent. Other demos: `demos.demo_davis` (DAVIS frames) and `demos.webcam_sam3`
(the SAM 3 teacher on a webcam, dev only).

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

**Without PyTorch:** the SAM-free pipeline exports to ONNX / ONNX Runtime Mobile. The exported models are in
the [`mobile-models-v1`](https://github.com/nakultt/EdgeVOS/releases/tag/mobile-models-v1) release. See
[`mobile/README.md`](mobile/README.md):

```bash
.venv/Scripts/python scripts/export_mobile.py --int8 --ort
```

```bash
.venv/Scripts/python -m mobile.demo --video samples/b.mp4 --prompt cat --int8
```

**Android:** install the APK from the [`android-v1.0`](https://github.com/nakultt/EdgeVOS/releases/tag/android-v1.0)
release, or build it from [`android/`](android). Open the app, point the camera at an object, type its name and
press Start. The first launch spends about 1.5 minutes compiling the models for the NPU; after that it starts in
under 2 s. The HUD shows fps, the backend (NPU/CPU) and per-stage times.

```bash
adb install -r EdgeVOS-android-1.0.apk
```

## Training (distillation)

1. Cache SAM 3.1 soft mask targets with COCO ground-truth boxes as prompts: `scripts/cache_sam3_targets.py`.
2. Distill RepViT-M1 + `SpatialMaskHead` on those targets: `python train/distill_mask.py --config-name base`.
3. Bundle the checkpoint: `scripts/bundle_checkpoint.py`. Export for mobile: `scripts/export_mobile.py`.
