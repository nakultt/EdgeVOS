"""
Build android/core/src/main/resources/scorer.onnx: the tracker's patch scoring
(`MobileTracker._scores`) as a tiny ONNX graph, so the ~0.6 GMAC per frame of
patch-vs-memory-bank similarities run in ONNX Runtime's SIMD matmul instead of scalar Kotlin.

    score = mean(topk(feats @ fg.T, k_fg)) - mean(topk(feats @ bg.T, k_bg))

Inputs: feats [N,D], fg [F,D], bg [B,D] float32; k_fg, k_bg [1] int64 (= min(knn, F), min(knn, B)).
Output: score [N].

    python android/tools/make_scorer.py
"""

import os

import onnx
from onnx import TensorProto, helper

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core", "src", "main", "resources", "scorer.onnx")


def main():
    f32, i64 = TensorProto.FLOAT, TensorProto.INT64
    inputs = [helper.make_tensor_value_info("feats", f32, ["N", "D"]),
              helper.make_tensor_value_info("fg", f32, ["F", "D"]),
              helper.make_tensor_value_info("bg", f32, ["B", "D"]),
              helper.make_tensor_value_info("k_fg", i64, [1]),
              helper.make_tensor_value_info("k_bg", i64, [1])]
    output = helper.make_tensor_value_info("score", f32, ["N"])
    axes = helper.make_tensor("axes", i64, [1], [1])
    nodes = []
    for bank in ("fg", "bg"):
        nodes += [
            helper.make_node("Gemm", ["feats", bank], [f"sim_{bank}"], transB=1),
            helper.make_node("TopK", [f"sim_{bank}", f"k_{bank}"], [f"top_{bank}", f"idx_{bank}"], axis=1),
            helper.make_node("ReduceMean", [f"top_{bank}", "axes"], [f"mean_{bank}"], keepdims=0),
        ]
    nodes.append(helper.make_node("Sub", ["mean_fg", "mean_bg"], ["score"]))
    graph = helper.make_graph(nodes, "edgevos_scorer", inputs, [output], initializer=[axes])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="edgevos")
    model.ir_version = 8
    onnx.checker.check_model(model)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    onnx.save(model, OUT)
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
