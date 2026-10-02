#!/usr/bin/env bash
# Benchmark the exported EdgeVOS models on a USB-connected Android phone.
#
#   bash mobile/android_bench/run.sh [cpu|qnn|all]
#
# Needs: adb, Android NDK (ANDROID_NDK or $ANDROID_HOME/ndk/<ver>), exported
# models in mobile/models (python scripts/export_mobile.py --int8 --ort).
# Downloads ONNX Runtime Android (+ QNN) and Qualcomm's QNN runtime from Maven
# Central into mobile/android_bench/build/. Everything on the phone goes to
# /data/local/tmp/edgevos and is removed at the end.
set -euo pipefail
export MSYS_NO_PATHCONV=1

MODE=${1:-all}
HERE=$(cd "$(dirname "$0")" && pwd)
HERE_SRC=$( (cygpath -m "$HERE" 2>/dev/null) || echo "$HERE")   # Windows NDK needs native paths
ROOT=$(cd "$HERE/../.." && pwd)
B=$HERE/build
D=/data/local/tmp/edgevos
ORT_VER=1.30.0          # CPU / XNNPACK / NNAPI
ORT_QNN_VER=1.29.0      # QNN execution provider (Qualcomm NPU)
QNN_RT_VER=2.42.0       # matches onnxruntime-android-qnn 1.29.0
MAVEN=https://repo1.maven.org/maven2

ADB=${ADB:-adb}
NDK=${ANDROID_NDK:-$(ls -d "${ANDROID_HOME:-$LOCALAPPDATA/Android/Sdk}"/ndk/* | sort -V | tail -1)}
CXX=$(ls "$NDK"/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android29-clang++* | head -1)

mkdir -p "$B" && cd "$B"
fetch() { [ -f "$2" ] || curl -sSfL -o "$2" "$1"; mkdir -p "${2%.aar}"; (cd "${2%.aar}" && unzip -q -o "../$2"); }
fetch "$MAVEN/com/microsoft/onnxruntime/onnxruntime-android/$ORT_VER/onnxruntime-android-$ORT_VER.aar" ort.aar
fetch "$MAVEN/com/microsoft/onnxruntime/onnxruntime-android-qnn/$ORT_QNN_VER/onnxruntime-android-qnn-$ORT_QNN_VER.aar" ort_qnn.aar
fetch "$MAVEN/com/qualcomm/qti/qnn-runtime/$QNN_RT_VER/qnn-runtime-$QNN_RT_VER.aar" qnn_rt.aar

"$CXX" -std=c++17 -O2 -DWITH_NNAPI "$HERE_SRC/bench.cpp" -Iort/headers -Lort/jni/arm64-v8a -lonnxruntime -static-libstdc++ -o bench_cpu
"$CXX" -std=c++17 -O2 "$HERE_SRC/bench.cpp" -Iort_qnn/headers -Lort_qnn/jni/arm64-v8a -lonnxruntime -static-libstdc++ -o bench_qnn

echo "Device: $($ADB shell getprop ro.product.model) / SoC $($ADB shell getprop ro.soc.model) / Android $($ADB shell getprop ro.build.version.release)"
HEX=$($ADB shell getprop ro.soc.model | tr -d '\r')
case "$HEX" in SM8650*) V=75 ;; SM8750*) V=79 ;; SM8550*) V=73 ;; SM8450*|SM8475*) V=69 ;; *) V=75 ;; esac

M=$( (cygpath -m "$ROOT/mobile/models" 2>/dev/null) || echo "$ROOT/mobile/models")
trap '$ADB shell "rm -rf $D"; echo "Removed $D from the device."' EXIT
$ADB shell "mkdir -p $D/cpu $D/qnn $D/models"
$ADB push bench_cpu ort/jni/arm64-v8a/libonnxruntime.so $D/cpu/ >/dev/null
$ADB push bench_qnn ort_qnn/jni/arm64-v8a/libonnxruntime.so qnn_rt/jni/arm64-v8a/libQnnHtp.so \
  qnn_rt/jni/arm64-v8a/libQnnHtpV${V}Stub.so qnn_rt/jni/arm64-v8a/libQnnHtpV${V}Skel.so \
  qnn_rt/jni/arm64-v8a/libQnnSystem.so qnn_rt/jni/arm64-v8a/libQnnHtpPrepare.so $D/qnn/ >/dev/null
$ADB push "$M/dinov3_patches.onnx" "$M/dinov3_patches.int8.onnx" "$M/segmenter.onnx" "$M/clip_dense.onnx" \
  "$M/clip_text.int8.onnx" $D/models/ >/dev/null
$ADB shell "chmod +x $D/cpu/bench_cpu $D/qnn/bench_qnn"

MODELS="segmenter:0:0:20 dinov3_patches:256:448:20 clip_dense:576:1024:5"
if [ "$MODE" != qnn ]; then
  echo "== CPU (4 threads) =="
  $ADB shell "cd $D/cpu && export LD_LIBRARY_PATH=$D/cpu;
    for m in $MODELS dinov3_patches.int8:256:448:20 clip_text.int8:0:0:10; do
      set -- \$(echo \$m | tr : ' '); n=\$1 h=\$2 w=\$3 it=\$4;
      ./bench_cpu ../models/\$n.onnx cpu \$h \$w \$it 4 2>&1 | tail -1; done"
fi
if [ "$MODE" != cpu ]; then
  echo "== QNN / Hexagon NPU (fp16, no CPU fallback, cached context) =="
  $ADB shell "cd $D/qnn && export LD_LIBRARY_PATH=$D/qnn NO_CPU_FALLBACK=1 ADSP_LIBRARY_PATH='$D/qnn;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp';
    for m in $MODELS; do
      set -- \$(echo \$m | tr : ' '); n=\$1 h=\$2 w=\$3 it=\$4;
      rm -f ../models/\${n}_qnn_ctx.onnx; CTX_OUT=../models/\${n}_qnn_ctx.onnx ./bench_qnn ../models/\$n.onnx qnn \$h \$w 1 4 2>&1 | tail -1 | sed 's/^/cold compile: /';
      ./bench_qnn ../models/\${n}_qnn_ctx.onnx qnn \$h \$w \$it 4 2>&1 | tail -1; done"
fi

