// Minimal ONNX Runtime latency benchmark for Android.
// usage: bench <model.onnx> <cpu|xnnpack|nnapi|qnn> <H> <W> <iters> <threads> [input.bin output.bin]
// input.bin: raw float32 for the (single) float input; output.bin: first output written as raw float32
#include <fstream>
#include <onnxruntime_cxx_api.h>
#ifdef WITH_NNAPI
#include <nnapi_provider_factory.h>
#endif
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

int main(int argc, char** argv) {
  if (argc < 7) { std::printf("usage: bench model ep H W iters threads\n"); return 1; }
  std::string model = argv[1], ep = argv[2];
  int64_t H = std::atoll(argv[3]), W = std::atoll(argv[4]);
  int iters = std::atoi(argv[5]), threads = std::atoi(argv[6]);

  Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "bench");
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(threads);
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (H > 0) { so.AddFreeDimensionOverrideByName("H", H); so.AddFreeDimensionOverrideByName("W", W); }
  if (const char* ctx = std::getenv("CTX_OUT")) {   // write a precompiled QNN context model
    so.AddConfigEntry("ep.context_enable", "1");
    so.AddConfigEntry("ep.context_file_path", ctx);
    so.AddConfigEntry("ep.context_embed_mode", "1");
  }
  if (std::getenv("NO_CPU_FALLBACK")) so.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
  if (ep == "xnnpack") {
    so.AppendExecutionProvider("XNNPACK", {{"intra_op_num_threads", std::to_string(threads)}});
  } else if (ep == "nnapi") {
#ifdef WITH_NNAPI
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(so, 0));
#endif
  } else if (ep == "qnn") {
    so.AppendExecutionProvider("QNN", {{"backend_path", "libQnnHtp.so"},
                                       {"htp_performance_mode", "burst"},
                                       {"enable_htp_fp16_precision", "1"}});
  }

  auto t0 = clk::now();
  Ort::Session sess(env, model.c_str(), so);
  double load = ms(t0, clk::now());

  Ort::AllocatorWithDefaultOptions alloc;
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<std::string> in_names, out_names;
  std::vector<Ort::Value> inputs;
  std::vector<std::vector<float>> fbufs;
  std::vector<std::vector<int64_t>> ibufs;
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> uni(0.f, 1.f);
  for (size_t i = 0; i < sess.GetInputCount(); ++i) {
    in_names.push_back(sess.GetInputNameAllocated(i, alloc).get());
    auto info = sess.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    for (size_t d = 0; d < shape.size(); ++d)
      if (shape[d] < 0) shape[d] = (d == 2 ? H : W);
    size_t n = 1;
    for (auto s : shape) n *= (size_t)s;
    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      ibufs.emplace_back(n, 0);
      auto& b = ibufs.back();
      b[0] = 49406; b[1] = 320; b[2] = 1125; b[3] = 539; b[4] = 320; b[5] = 2368; b[6] = 269; b[7] = 49407;
      inputs.push_back(Ort::Value::CreateTensor<int64_t>(mem, b.data(), n, shape.data(), shape.size()));
    } else {
      fbufs.emplace_back(n);
      if (argc > 8) {
        std::ifstream f(argv[7], std::ios::binary);
        f.read(reinterpret_cast<char*>(fbufs.back().data()), n * sizeof(float));
      } else {
        for (auto& v : fbufs.back()) v = uni(rng);
      }
      inputs.push_back(Ort::Value::CreateTensor<float>(mem, fbufs.back().data(), n, shape.data(), shape.size()));
    }
  }
  for (size_t i = 0; i < sess.GetOutputCount(); ++i) out_names.push_back(sess.GetOutputNameAllocated(i, alloc).get());
  std::vector<const char*> in_c, out_c;
  for (auto& s : in_names) in_c.push_back(s.c_str());
  for (auto& s : out_names) out_c.push_back(s.c_str());

  for (int i = 0; i < 3; ++i) sess.Run(Ort::RunOptions{nullptr}, in_c.data(), inputs.data(), inputs.size(), out_c.data(), out_c.size());
  std::vector<double> t;
  for (int i = 0; i < iters; ++i) {
    auto a = clk::now();
    sess.Run(Ort::RunOptions{nullptr}, in_c.data(), inputs.data(), inputs.size(), out_c.data(), out_c.size());
    t.push_back(ms(a, clk::now()));
  }
  if (argc > 8) {
    auto outs = sess.Run(Ort::RunOptions{nullptr}, in_c.data(), inputs.data(), inputs.size(), out_c.data(), out_c.size());
    auto n = outs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::ofstream f(argv[8], std::ios::binary);
    f.write(reinterpret_cast<const char*>(outs[0].GetTensorData<float>()), n * sizeof(float));
  }
  std::sort(t.begin(), t.end());
  double mean = 0;
  for (double v : t) mean += v;
  mean /= t.size();
  std::printf("RESULT model=%s ep=%s threads=%d load_ms=%.0f mean_ms=%.2f median_ms=%.2f min_ms=%.2f\n",
              model.c_str(), ep.c_str(), threads, load, mean, t[t.size() / 2], t[0]);
  return 0;
}
