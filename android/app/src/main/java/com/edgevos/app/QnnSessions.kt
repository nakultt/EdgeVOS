package com.edgevos.app

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.Build
import android.util.Log
import com.edgevos.core.CpuSessions
import com.edgevos.core.Manifest
import com.edgevos.core.SessionProvider
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * Vision models on the Hexagon NPU through the ONNX Runtime QNN EP (fp16, burst), with compiled
 * contexts cached in app storage; the INT8 text encoder on the CPU; and a CPU-EP fallback for any
 * model QNN cannot take.
 *
 * The first QNN load of a model+shape compiles the graph (CLIP ~30 s, DINOv3 ~4 s, segmenter ~1.6 s)
 * and dumps `<model>_<H>x<W>_qnn_ctx.onnx`; later launches load that in 0.2–0.5 s.
 */
class QnnSessions(
    context: Context,
    private val dir: File,
    private val manifest: Manifest,
    private val useNpu: Boolean = true,
) : SessionProvider {
    private val env = OrtEnvironment.getEnvironment()
    private val backends = ConcurrentHashMap<String, String>()
    /** Context binaries are SoC- and build-specific: key the cache by both. */
    val ctxDir = File(context.filesDir, "qnn_ctx/${socModel()}-${appVersion(context)}").apply { mkdirs() }
    private val cpu = CpuSessions(dir, manifest, threads = 4, int8Models = setOf("clip_text", "dinov3_patches"))

    override fun backend(model: String) = backends[model] ?: "?"

    /** Keyed by shape and model file size, so re-exported models are recompiled. */
    fun ctxFile(model: String, h: Int, w: Int) =
        File(ctxDir, "${model}_${h}x${w}_${File(dir, manifest.file(model)).length()}_qnn_ctx.onnx")

    override fun create(model: String, h: Int, w: Int): OrtSession {
        if (model == "clip_text" || !useNpu) return cpu(model, h, w)
        val ctx = ctxFile(model, h, w)
        val t0 = System.nanoTime()
        try {
            val session = if (ctx.exists()) {
                try {
                    env.createSession(ctx.path, qnnOptions(h, w))
                } catch (e: Exception) {
                    Log.w(TAG, "cached context $ctx failed to load, recompiling", e)
                    ctx.delete()
                    compile(model, h, w, ctx)
                }
            } else {
                compile(model, h, w, ctx)
            }
            Log.i(TAG, "QNN session $model ${h}x$w ready in ${(System.nanoTime() - t0) / 1_000_000} ms")
            backends[model] = "NPU"
            return session
        } catch (e: Exception) {
            Log.w(TAG, "QNN failed for $model ${h}x$w, falling back to the CPU EP", e)
            return cpu(model, h, w)
        }
    }

    private fun cpu(model: String, h: Int, w: Int): OrtSession {
        val t0 = System.nanoTime()
        return cpu.create(model, h, w).also {
            backends[model] = "CPU"
            Log.i(TAG, "CPU session $model ${h}x$w ready in ${(System.nanoTime() - t0) / 1_000_000} ms")
        }
    }

    private fun compile(model: String, h: Int, w: Int, ctx: File): OrtSession {
        val tmp = File(ctxDir, "${model}_${h}x${w}_tmp_qnn_ctx.onnx").apply { delete() }
        val so = qnnOptions(h, w).apply {
            addConfigEntry("ep.context_enable", "1")
            addConfigEntry("ep.context_embed_mode", "1")
            addConfigEntry("ep.context_file_path", tmp.path)
        }
        Log.i(TAG, "compiling $model ${h}x$w for the NPU (one time)")
        val session = env.createSession(File(dir, manifest.file(model)).path, so)
        if (!tmp.renameTo(ctx)) Log.w(TAG, "could not cache the QNN context for $model at $ctx")
        return session
    }

    private fun qnnOptions(h: Int, w: Int) = CpuSessions.cpuOptions(1).apply {
        // Run all of the graph on the NPU or fail (then the whole model falls back to the CPU EP).
        addConfigEntry("session.disable_cpu_ep_fallback", "1")
        addQnn(mapOf(
            "backend_path" to "libQnnHtp.so",
            "htp_performance_mode" to "burst",
            "enable_htp_fp16_precision" to "1",
        ))
        if (h > 0) { setSymbolicDimensionValue("H", h.toLong()); setSymbolicDimensionValue("W", w.toLong()) }
    }

    companion object {
        private const val TAG = "EdgeVOS"

        fun socModel(): String = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE

        private fun appVersion(context: Context): Long =
            context.packageManager.getPackageInfo(context.packageName, 0).longVersionCode
    }
}
