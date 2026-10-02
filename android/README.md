# EdgeVOS for Android

Text-prompted object segmentation and tracking, on-device. This is the Kotlin port of
[`mobile/runtime.py`](../mobile/runtime.py) (the reference pipeline) and the app around it.

| Module | What it is |
|---|---|
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

Parity: the same frames go through both trackers. Both decode b.mp4 with the ffmpeg CLI to rgb24.
The model setup matches the app: fp32 vision models and the INT8 text encoder, on the ORT CPU EP.

```bash
pip install -r mobile/requirements.txt
python android/tools/parity_reference.py --video b.mp4 --prompt cat --out parity/py_cat
(cd android && ./gradlew :core:parity --args="--video ../b.mp4 --prompt cat --out ../parity/kt_cat")
python android/tools/parity_compare.py parity/py_cat parity/kt_cat
```

Results on `b.mp4` (944 frames, 852×480):

| Prompt | Mean mask IoU, Kotlin vs Python | Events |
|---|---|---|
| `cat` | **0.9995**: 929/929 frames with a mask in both, worst frame 0.90 (frames 503–508, just after a re-ground) | identical: `init`; `reground:scene_cut` at 126, 288, 412, 503, 708; cut at 841 → `absent` (841, 846, 851); `reground:absent` at 856 |
| `car` | **1.0000**: absent on all 944 frames in both | identical: `absent` retries only |

At 841 the scene cut is detected, but grounding doesn't confirm the cat, so the frame is `absent`. The
re-acquisition is confirmed at 856. Python behaves the same with the fp32 text encoder.

Speed on the desktop JVM (4-core Xeon, CPU EP): 143 ms per tracked frame (DINOv3 110, segmenter 20,
scoring 3.5, Kotlin glue ~9). Python takes 185 ms per tracked frame on the same machine.
