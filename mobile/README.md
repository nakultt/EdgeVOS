# EdgeVOS on device

The product pipeline is SAM-free and runs on [ONNX Runtime Mobile](https://onnxruntime.ai/docs/tutorials/mobile/) (Android and iOS).
`mobile/runtime.py` is the reference implementation, built on ONNX Runtime + numpy + OpenCV with no PyTorch. A mobile app ports it file for file.

## 1. Export the models

```bash
.venv/Scripts/python scripts/export_mobile.py --int8 --ort
```

This writes `mobile/models/`:

| File | Input → output | Size | Runs |
|---|---|---|---|
| `dinov3_patches` | image `[1,3,H,W]` RGB 0..1, H/W multiples of 16 → patches `[H/16·W/16, 384]` | 82.7 MB, INT8 21.3 MB | every frame |
| `segmenter` | crop `[1,3,224,224]` → probs `[1,1,256,256]` | 23.0 MB | every frame + grounding |
| `clip_dense` | image `[1,3,H,W]`, H/W multiples of 32 → dense `[H/32,W/32,512]` | 136.9 MB | scene cuts / re-acquisition |
| `clip_text` | tokens `[1,77]` int64 → `[1,512]` | 242 MB, INT8 61.2 MB | once per prompt |
| `clip_background.npy` | 10 background prompt embeddings | 20 KB | grounding |
| `manifest.json` | shapes, strides, long sides, tracker config, thresholds | | |

Each model has `.onnx` and `.ort` versions. `required_operators.config` lets you build a reduced-size ONNX Runtime.
Normalisation, token slicing, L2-normalisation and the sigmoid are inside the graphs.

## 2. Check it on desktop, without PyTorch

```bash
.venv/Scripts/python -m mobile.demo --video b.mp4 --prompt cat --int8
```

## 3. Per-frame pipeline (what the app implements)

| Step | Code (`runtime.py`) | Notes |
|---|---|---|
| Resize the frame so its long side is 448, keeping aspect, multiples of 16 → `dinov3_patches` | `Models.dino_patches` | sizes via `Models.input_size` |
| Scene cut: grey thumbnail 64×36 diff vs recent median, plus fraction of patches with cos < 0.5 | `_scene_cut` | |
| Patch score = top-5 mean sim to fg bank − top-5 mean sim to bg bank | `_scores` | one matmul each, banks ≤ 1.5k × 384 |
| Motion window around the previous box → connected component overlapping it | `_propagate` | grid is ~28×16 |
| Crop the component's box +15% → resize 224 → `segmenter` → paste, gated by the dilated component | `_mask_probs` | |
| Re-ground on cut, after 3 low-confidence frames (10-frame cooldown), or every 5 frames while absent | `_reground` | |
| Grounding: long side 1024 → `clip_dense` → `dense·text − max(dense·background)` → peaks → boxes; mask = segmenter iterated 3× on its own box | `Grounder.detect`, `_detection_mask` | |
| Memory: anchor from the grounded frame + FIFO of 4 keyframes, written only when confident and area-stable | `_commit`, `PatchMemory` | |

The tokenizer is `mobile/tokenizer.py`, a CLIP BPE port with the vocabulary in `mobile/assets/`. It's only needed when the prompt changes; you can also precompute embeddings for a fixed prompt list and drop `clip_text`.

## 4. ONNX Runtime Mobile notes

- **Android:** `com.microsoft.onnxruntime:onnxruntime-android` (or `-mobile` for ORT format), with execution providers `NNAPI` / `QNN` / `XNNPACK`.
- **iOS:** the `onnxruntime-objc` pod, with the `CoreML` execution provider.
- **Fixed shapes for NPUs:** image inputs have dynamic `H`/`W`. NPU execution providers want static shapes, so fix them per session with free-dimension overrides, e.g. `SessionOptions.addFreeDimensionOverrideByName("H", 256)`. For the common 16:9 landscape case that's 256×448 (DINOv3) and 576×1024 (grounding); create a second session for portrait.
- **Threading:** disable intra-op spinning (`session.intra_op.allow_spinning = 0`). The models run back to back with other work in between, and spinning threads made DINOv3 4× slower on CPU in testing.

## 5. Measured so far (desktop CPU stand-in)

On a laptop CPU (ONNX Runtime CPU EP, no GPU): ~125 ms per tracked frame (DINOv3 72 ms, segmenter 26 ms, glue ~27 ms).
Per-frame compute is ~24 GFLOPs (DINOv3 19.5 + segmenter 3.4 + kNN ~1.5), and grounding is ~142 GFLOPs per call.
Real phone numbers need a device: see issue #21.
