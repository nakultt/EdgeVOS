# EdgeVOS for Android

Text-prompted object segmentation and tracking, on-device. This is the Kotlin port of
[`mobile/runtime.py`](../mobile/runtime.py) (the reference pipeline) and the app around it.

| Module | What it is |
|---|---|
| `app/` | The Android app: CameraX preview, prompt box, mask overlay, HUD; QNN (Hexagon NPU) sessions with cached contexts; on-device parity mode. |
| `core/` | Plain Kotlin/JVM library: CLIP tokenizer, OpenCV-equivalent image ops, `Grounder` / `PatchMemory` / `MobileTracker`, ONNX Runtime model wrapper. It has no Android dependencies, so it is unit-tested and parity-checked on the desktop JVM. |
| `tools/` | Python helpers: reference dumps from `mobile/runtime.py`, mask comparison, test fixtures, scorer graph. |

## The tracker port (`core/`)

`core/src/main/kotlin/com/edgevos/core/` maps file for file onto the Python reference:

| Kotlin | Python |
|---|---|
| `ClipTokenizer.kt` | `mobile/tokenizer.py` (vocabulary `mobile/assets/bpe_simple_vocab_16e6.txt.gz`, packaged as a resource) |
| `ImageOps.kt` | `resize` (OpenCV `INTER_LINEAR` / `INTER_AREA`), `cv2.dilate`, `cv2.connectedComponents`, `grid_fraction`, `bbox`, `box_mask`, `box_iou` |
| `Models.kt` | `Models` (sessions, input sizes, resizing to NCHW) + `manifest.json` |
| `MobileTracker.kt` | `Grounder`, `PatchMemory`, `MobileTracker` |
| `NumpyRandom.kt` | `np.random.default_rng(0).permutation` (bit-exact PCG64), used by `PatchMemory` bank subsampling |
| `resources/scorer.onnx` | `MobileTracker._scores` as a 5-node ONNX graph (`tools/make_scorer.py`) |

Why the patch scoring runs in ONNX Runtime: the similarities to the memory banks cost about 0.6 GMAC per
frame (448 patches × up to ~3.7k bank rows × 384). In scalar Kotlin that is hundreds of ms on a phone.
ORT's MLAS matmul runs it in a few ms (3.5 ms on the desktop JVM). The bank tensors are re-uploaded only
when the memory changes.

Numerics follow the NumPy 2 reference. Boxes are float32, and comparisons of float32 arrays against
Python floats are done in float32 (NEP 50). Resizes reproduce OpenCV's float code path.

### Tests and the JVM parity check

Needs JDK 17+ and the models in `mobile/models/`. Download them from the `mobile-models-v1` release:

```bash
gh release download mobile-models-v1 --repo nakultt/EdgeVOS --dir mobile/models
```

If `gh` can't use GraphQL, download each asset through the REST API instead:

```bash
gh api repos/nakultt/EdgeVOS/releases/tags/mobile-models-v1 --jq '.assets[] | "\(.id) \(.name)"' |
  while read id name; do gh api -H "Accept: application/octet-stream" repos/nakultt/EdgeVOS/releases/assets/$id > mobile/models/$name; done
(cd mobile/models && sha256sum -c SHA256SUMS)
```

Unit tests compare the tokenizer ids, the numpy RNG stream and permutations, the resizes, connected
components and grid fractions against golden values from Python (`tools/make_fixtures.py`):

```bash
cd android && ./gradlew :core:test
```

Parity: the same frames go through both trackers. Both decode the video with the ffmpeg CLI to rgb24.
The model setup matches the app: fp32 vision models and the INT8 text encoder, on the ORT CPU EP. Run the
Python reference on the app's ONNX Runtime version (1.29.0, `tools/requirements.txt`). The INT8 text
encoder's output changes between ORT releases (1.29 vs 1.30: cosine 0.9989).

```bash
pip install -r android/tools/requirements.txt
python android/tools/parity_reference.py --video b.mp4 --prompt cat --out parity/py_cat
(cd android && ./gradlew :core:parity --args="--video ../b.mp4 --prompt cat --out ../parity/kt_cat")
python android/tools/parity_compare.py parity/py_cat parity/kt_cat
```

Results, Kotlin on the JVM vs `mobile/runtime.py`, both on ORT 1.29.0:

| Clip | Prompt | Mean mask IoU | Events |
|---|---|---|---|
| `b.mp4` (944 frames, 852×480) | `cat` | **1.0000**: 931/944 frames bit-identical, worst 0.99994 | identical: `init`; `reground:scene_cut` at 126, 288, 412, 503, 708; cut at 841 → `absent` (841, 846, 851); `reground:absent` at 856 |
| `b.mp4` | `car` | **1.0000**: absent on all 944 frames in both | identical: `absent` retries only |
| `b.mp4` centre 9:16 crop at 360×640 (400 frames, the app's frame size: DINOv3 448×256, CLIP 1024×576) | `cat` | **1.0000** | identical, including a `reground:low_confidence` at 156 |

At 841 the scene cut is detected, but grounding doesn't confirm the cat, so the frame is `absent`. The
re-acquisition is confirmed at 856. Python behaves the same with the fp32 text encoder.

With the Python reference on ORT 1.30.0 instead, b.mp4 `cat` gives 0.9995 with identical events. Grounding is
sensitive to small embedding changes. On one frame, a 1-px change in the grounding box changes the
3× refined detection mask by tens of percent. Expect the NPU's fp16 to cause similar divergences on
some clips.

Speed on the desktop JVM (4-core Xeon, CPU EP): 143 ms per tracked frame (DINOv3 110, segmenter 20,
scoring 3.5, Kotlin glue ~9). Python takes 185 ms per tracked frame on the same machine.

## The app (`app/`)

Full-screen back-camera preview, a text box and a Start/Stop button. Start grounds the prompt on the
current frame, then tracks every frame. Submitting a different prompt restarts tracking. The mask is drawn
at 50% opacity: green on (re-)grounded frames, red on tracked frames, nothing while the object is
absent. The HUD shows fps, per-frame ms, the event (`init`, `reground:<reason>`, `track`, `absent`), the
backend (`NPU` / `CPU`) and per-stage times. Every 30th frame and every event is also logged under the
`EdgeVOS` logcat tag.

**Frame path** (`MainActivity`, `YuvConverter`, `Engine`):
- CameraX `ImageAnalysis` at 1280×720 16:9 with `STRATEGY_KEEP_ONLY_LATEST`, in `YUV_420_888`.
- One pass converts to RGB (full-range BT.601), applies a 2×2 box downscale and rotates by `rotationDegrees`, giving a 360×640 upright portrait frame.
- The tracker runs on the analysis thread, so frames that arrive while it is busy are dropped.
- Model inputs are DINOv3 448×256 and CLIP 1024×576. These are the measured sizes.
- The overlay maps the 360×640 mask with the same centre-crop as `PreviewView`'s `FILL_CENTER`, so it lines up with the preview. Preview and analysis share the 16:9 aspect.

**Inference backend** (`QnnSessions`, `Engine`):
- **Libraries:** `onnxruntime-android-qnn:1.29.0` and `qnn-runtime:2.42.0`.
  - The APK keeps only `libQnnHtp.so`, `libQnnHtpV75Stub.so`, `libQnnHtpV75Skel.so`, `libQnnSystem.so` and `libQnnHtpPrepare.so`. The other Hexagon versions and the DSP/GPU backends are excluded in `app/build.gradle.kts`.
  - `useLegacyPackaging` sets `extractNativeLibs=true`.
  - `ADSP_LIBRARY_PATH` points the DSP loader at the extracted libraries.
  - The manifest declares `libcdsprpc.so` as a `uses-native-library`.
- **Vision models** (`dinov3_patches`, `segmenter`, `clip_dense`) run on the QNN EP:
  - Options: `backend_path=libQnnHtp.so`, `htp_performance_mode=burst`, `enable_htp_fp16_precision=1`.
  - `H`/`W` are fixed per session with free-dimension overrides.
  - `session.disable_cpu_ep_fallback=1` makes a partly-placed graph fail outright instead of silently mixing NPU and CPU.
- **Context cache:** the first QNN load of each model+shape compiles the graph and writes `filesDir/qnn_ctx/<SoC>-<versionCode>/<model>_<H>x<W>_<size>_qnn_ctx.onnx`. It uses `ep.context_enable=1` and `ep.context_embed_mode=1`. Later launches load the cached context.
  - The first launch shows "Optimizing models for this phone…" while it compiles the portrait sessions (segmenter, DINOv3 448×256, CLIP 1024×576).
  - Other shapes, such as landscape for the parity video, compile lazily on first use.
- **CPU fallback:** if QNN fails for a model, that model runs on the ORT CPU EP (4 threads, DINOv3 INT8). The HUD then shows `CPU` (or `NPU+CPU`).
- **Text encoder:** `clip_text.int8` always runs on the CPU, once per prompt.
- **Patch scorer** (`scorer.onnx`): runs on the CPU.
- **Spinning:** intra-op spinning is disabled everywhere.

### Models on the phone

**Chosen approach: bundled in the APK.** `app/build.gradle.kts` (`prepareModelAssets`) copies the six files
the app uses into `assets/models/` uncompressed (`noCompress`):
- `dinov3_patches.onnx` (NPU) and `dinov3_patches.int8.onnx` (CPU fallback)
- `segmenter.onnx`
- `clip_dense.onnx`
- `clip_text.int8.onnx`
- `manifest.json`

It also writes `clip_background.npy` as a raw float32 file, `clip_background.f32`. On first launch the app copies these to
`filesDir/models`, because ONNX Runtime and QNN need real files. The debug APK is about 406 MB, and one `adb install` gives a
self-contained app.

Alternative for faster iteration: build with `-PbundleModels=false` for a small APK, then push the
models once. The app uses `<external files>/models` whenever `manifest.json` is there.

```bash
adb shell mkdir -p /sdcard/Android/data/com.edgevos.app/files/models
adb push mobile/models/{manifest.json,dinov3_patches.onnx,dinov3_patches.int8.onnx,segmenter.onnx,clip_dense.onnx,clip_text.int8.onnx,clip_background.npy} \
  /sdcard/Android/data/com.edgevos.app/files/models/
```

### Build

Needs JDK 17+, Android SDK platform 36 and build-tools 36.0.0 (`sdkmanager "platforms;android-36"
"build-tools;36.0.0" "platform-tools"`), and the models in `mobile/models/` (see above). No NDK: the app has no
native code of its own.

```bash
cd android
echo "sdk.dir=$ANDROID_HOME" > local.properties     # Windows: sdk.dir=C\:\\Users\\<you>\\AppData\\Local\\Android\\Sdk
./gradlew :core:test :app:testDebugUnitTest :app:assembleDebug
# → app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.edgevos.app/.MainActivity
adb logcat -s EdgeVOS                              # startup time, backend per model, per-frame stages
```

### On-device parity

`ParityActivity` runs the tracker (NPU sessions, landscape shapes) on a video's frames. It writes the same
`masks.bin` / `events.tsv` as the JVM and Python runs. Use raw rgb24 frames to give it byte-identical input:

```bash
ffmpeg -i b.mp4 -f rawvideo -pix_fmt rgb24 b.rgb                      # 1.16 GB
adb push b.rgb /sdcard/Android/data/com.edgevos.app/files/
adb shell am start -n com.edgevos.app/.ParityActivity --es prompt cat \
  --es raw /sdcard/Android/data/com.edgevos.app/files/b.rgb --ei width 852 --ei height 480
adb logcat -s EdgeVOS-Parity                                           # wait for DONE
adb pull /sdcard/Android/data/com.edgevos.app/files/parity/cat parity/device_cat
python android/tools/parity_compare.py parity/py_cat parity/device_cat
```

`--es video <mp4 on the phone>` decodes on the phone instead. Its YUV→RGB differs slightly from ffmpeg's.
Repeat with `--es prompt car`, which must stay absent on every frame.

### Expected and measured speed

The models are measured on a Galaxy S24 NPU (`mobile/README.md`): DINOv3 9.2 ms and segmenter 1.5 ms per frame,
CLIP grounding 50 ms at scene cuts. On the desktop JVM at 852×480, scoring takes 3.5 ms and the rest of the
Kotlin glue around the models (resizes, masks, components) about 9 ms. At the app's 360×640 frames the glue is
smaller. Adding an estimated ~5 ms for YUV conversion budgets roughly 30 fps end to end on the NPU. That is an estimate, not a measurement; the target is ≥ 15 fps.

**On-device measurements are pending.** They need the phone: fps, per-stage ms, first- and second-launch
startup, and parity vs Python. See the checklist in the app PR.
