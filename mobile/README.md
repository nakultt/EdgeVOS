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

## 5. Measured on a phone: Galaxy S24 (Snapdragon 8 Gen 3, SM8650, Android 16)

Reproduce with `bash mobile/android_bench/run.sh` (adb + Android NDK). The script removes everything it puts on the device.

| Model | Phone CPU (ORT CPU EP, 4 threads) | **Hexagon NPU (QNN EP, fp16)** | Runs |
|---|---|---|---|
| `dinov3_patches` 256×448 | 196 ms (INT8: 86 ms) | **9.2 ms** | every frame |
| `segmenter` (distilled RepViT) | 29 ms | **1.5 ms** | every frame + grounding |
| `clip_dense` 576×1024 | 1,606 ms | **50 ms** | scene cuts / re-acquisition |
| `clip_text` (INT8) | 15 ms | (CPU) | once per prompt |

- **Per-frame model time on the NPU is ~11 ms**, about 90 fps of model budget before app glue. Grounding at a scene cut costs ~50 ms plus 3 segmenter passes.
- **All three vision models run 100% on the NPU:** session creation succeeds with `session.disable_cpu_ep_fallback = 1`.
- **NPU fp16 matches CPU fp32 on real frames:** segmenter mask IoU 0.989–0.994, DINOv3 patch cosine 0.998 mean, CLIP dense cosine 0.992 mean.
- **Sustained load:** 3,000 consecutive DINOv3 runs averaged 9.5 ms vs 9.2 ms best, so no thermal throttling within ~30 s (chip 51 → 70 °C).
- **Startup:** the first NPU load compiles the graph (segmenter 1.6 s, DINOv3 4.3 s, CLIP 30 s). Save the compiled context (`ep.context_enable=1`, `ep.context_embed_mode=1`) on first launch and load `*_qnn_ctx.onnx` afterwards: **0.2–0.5 s** per model. Context binaries are SoC-specific, so generate them on device or ship one per Hexagon version.
- **NPU-ready bundle (cached fp16 contexts):** segmenter 13 MB + DINOv3 46 MB + CLIP 86 MB + text INT8 61 MB ≈ **206 MB**.

**Android setup that was measured:**
- **Libraries:** `com.microsoft.onnxruntime:onnxruntime-android-qnn:1.29.0` + `com.qualcomm.qti:qnn-runtime:2.42.0` (`libQnnHtp.so`, `libQnnHtpV75Stub/Skel.so`, `libQnnSystem.so`).
- **QNN EP options:** `backend_path=libQnnHtp.so`, `htp_performance_mode=burst`, `enable_htp_fp16_precision=1`.
- **Shapes:** free-dimension overrides `H`/`W` per orientation.
- **CPU fallback:** use the CPU EP; XNNPACK was slower here (DINOv3 INT8 147 ms vs 86 ms).
