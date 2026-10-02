package com.edgevos.app

import android.content.Context
import android.system.Os
import android.util.Log
import com.edgevos.core.FloatImage
import com.edgevos.core.Manifest
import com.edgevos.core.Mask
import com.edgevos.core.MobileTracker
import com.edgevos.core.Models
import java.io.File

/** One tracker result for the UI. */
class FrameResult(
    val mask: Mask?,
    /** "init", "reground:<reason>", "track" or "absent". */
    val event: String,
    val grounded: Boolean,
    val trackMs: Double,
    val stages: String,
)

/**
 * Process-wide models + tracker. Loaded once on a background thread; [process] runs on the camera
 * analysis thread (or the parity thread), never concurrently.
 */
object Engine {
    private const val TAG = "EdgeVOS"

    /** Camera frames are 16:9 portrait, downscaled to this long side before tracking. */
    const val FRAME_LONG_SIDE = 640

    @Volatile var models: Models? = null
        private set
    @Volatile var tracker: MobileTracker? = null
        private set
    var backendLabel = "?"
        private set
    var startupMs = 0L
        private set

    /** True when no compiled QNN context is cached yet: the first launch compiles them (slow). */
    fun needsOptimizing(context: Context): Boolean {
        val dir = context.filesDir.listFiles()?.firstOrNull { it.name == "qnn_ctx" } ?: return true
        return dir.walkTopDown().none { it.name.endsWith("_qnn_ctx.onnx") && !it.name.contains("_tmp_") }
    }

    @Synchronized
    fun load(context: Context, progress: (String) -> Unit) {
        if (models != null) return
        val t0 = System.nanoTime()
        // Let the QNN HTP stub find the V75 skel shipped in the APK's (extracted) native lib dir.
        val libDir = context.applicationInfo.nativeLibraryDir
        Os.setenv("ADSP_LIBRARY_PATH", "$libDir;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/vendor/dsp", true)

        val dir = ModelStore.resolve(context, progress)
        val manifest = Manifest.load(File(dir, "manifest.json"))
        val bg = File(dir, "clip_background.f32").let { if (it.exists()) it else File(dir, "clip_background.npy") }
        val sessions = QnnSessions(context, dir, manifest)
        val m = Models(manifest, sessions, Models.loadBackground(bg))

        // Create the portrait sessions the camera uses up front (compiles + caches them on first launch).
        val (dh, dw) = m.inputSize("dinov3_patches", FRAME_LONG_SIDE, FRAME_LONG_SIDE * 9 / 16)
        val (ch, cw) = m.inputSize("clip_dense", FRAME_LONG_SIDE, FRAME_LONG_SIDE * 9 / 16)
        for ((label, create) in listOf<Pair<String, () -> Unit>>(
            "segmenter" to { m.session("segmenter") },
            "DINOv3 ${dh}x$dw" to { m.session("dinov3_patches", dh, dw) },
            "CLIP ${ch}x$cw" to { m.session("clip_dense", ch, cw) },
            "text encoder" to { m.session("clip_text") },
        )) {
            progress("Preparing $label")
            create()
        }
        val npu = listOf("dinov3_patches", "segmenter", "clip_dense").map { m.backend(it) }
        backendLabel = when {
            npu.all { it == "NPU" } -> "NPU"
            npu.none { it == "NPU" } -> "CPU"
            else -> "NPU+CPU"
        }
        tracker = MobileTracker(m)
        models = m
        startupMs = (System.nanoTime() - t0) / 1_000_000
        Log.i(TAG, "startup: models ready in $startupMs ms, backend $backendLabel " +
            "(dino=${m.backend("dinov3_patches")} seg=${m.backend("segmenter")} clip=${m.backend("clip_dense")} text=${m.backend("clip_text")})")
    }

    /** Ground [prompt] on this frame when [initPrompt] is set, otherwise track. */
    fun process(frame: FloatImage, initPrompt: String?): FrameResult {
        val t = tracker!!
        val t0 = System.nanoTime()
        val mask = if (initPrompt != null) t.init(frame, initPrompt) else t.step(frame)
        val ms = (System.nanoTime() - t0) / 1e6
        val ev = t.lastEvent
        val grounded = ev != null && (ev.startsWith("init") || ev.startsWith("reground"))
        val event = ev ?: if (mask == null) "absent" else "track"
        return FrameResult(mask, event, grounded && mask != null, ms, models!!.timings.toString())
    }
}
