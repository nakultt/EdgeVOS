package com.edgevos.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/** mobile/models/manifest.json */
class Manifest(json: String) {
    @Suppress("UNCHECKED_CAST")
    private val root = Json.parse(json) as Map<String, Any?>

    @Suppress("UNCHECKED_CAST")
    val trackerCfg = root["tracker_cfg"] as Map<String, Any?>

    @Suppress("UNCHECKED_CAST")
    val models = root["models"] as Map<String, Map<String, Any?>>

    @Suppress("UNCHECKED_CAST")
    private val grounding = root["grounding"] as Map<String, Any?>
    val presenceThreshold = (grounding["presence_threshold"] as Double).toFloat()
    val promptTemplate = grounding["prompt_template"] as String

    fun stride(model: String) = (models.getValue(model)["stride"] as Double).toInt()
    fun longSide(model: String) = (models.getValue(model)["long_side"] as Double).toInt()
    fun file(model: String, int8: Boolean = false): String {
        val m = models.getValue(model)
        return (if (int8) m["int8_file"] as String? else null) ?: m["file"] as String
    }

    companion object {
        fun load(file: File) = Manifest(file.readText())
    }
}

/** manifest `tracker_cfg` (see MobileTracker in mobile/runtime.py). */
class TrackerConfig(private val m: Map<String, Any?>) {
    private fun d(k: String) = (m[k] as Number).toDouble()
    private fun i(k: String) = (m[k] as Number).toInt()
    val knn = i("knn")
    val maxFgBank = i("max_fg_bank")
    val maxBgBank = i("max_bg_bank")
    val recentKeyframes = i("recent_keyframes")
    val keyframeInterval = i("keyframe_interval")
    val keyframeMinConf = d("keyframe_min_conf")
    val searchMargin = d("search_margin").toFloat()
    val lowConf = d("low_conf")
    val lowConfFrames = i("low_conf_frames")
    val regroundCooldown = i("reground_cooldown")
    val regroundInterval = i("reground_interval")
    val absentRetry = i("absent_retry")
    val cutGrayDiff = d("cut_gray_diff")
    val cutGrayRatio = d("cut_gray_ratio")
    val cutPatchFrac = d("cut_patch_frac")
    val minDetAppearance = d("min_det_appearance")
    val maskThreshold = d("mask_threshold").toFloat()
    val boxRefineIters = i("box_refine_iters")
}

/** Creates ONNX Runtime sessions; [h]/[w] are the free-dimension overrides for image models (0 = none). */
interface SessionProvider : AutoCloseable {
    fun create(model: String, h: Int, w: Int): OrtSession

    /** "NPU", "CPU", ... for the HUD. */
    fun backend(model: String): String = "CPU"
    override fun close() {}
}

/** ORT CPU EP sessions straight from the model directory (desktop parity runs and the app's fallback). */
class CpuSessions(
    private val dir: File,
    private val manifest: Manifest,
    private val threads: Int = 0,
    private val int8Models: Set<String> = setOf("clip_text"),
) : SessionProvider {
    private val env = OrtEnvironment.getEnvironment()

    override fun create(model: String, h: Int, w: Int): OrtSession {
        val so = cpuOptions(threads)
        if (h > 0) { so.setSymbolicDimensionValue("H", h.toLong()); so.setSymbolicDimensionValue("W", w.toLong()) }
        return env.createSession(File(dir, manifest.file(model, model in int8Models)).path, so)
    }

    companion object {
        fun cpuOptions(threads: Int = 0) = OrtSession.SessionOptions().apply {
            // Models run back to back with tracker work in between; spinning threads would contend with it.
            addConfigEntry("session.intra_op.allow_spinning", "0")
            if (threads > 0) setIntraOpNumThreads(threads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
    }
}

/** Per-frame stage times in ms (summed when a model runs more than once in a frame). */
class Timings {
    val ms = LinkedHashMap<String, Double>()
    fun reset() = ms.clear()
    fun add(stage: String, t: Double) { ms[stage] = (ms[stage] ?: 0.0) + t }
    inline fun <T> time(stage: String, block: () -> T): T {
        val t0 = System.nanoTime()
        try { return block() } finally { add(stage, (System.nanoTime() - t0) / 1e6) }
    }
    override fun toString() = ms.entries.joinToString(" ") { "${it.key}=%.1f".format(it.value) }
}

/** DINOv3 patch features: [gh*gw, dim] rows, L2-normalised. */
class PatchFeats(val data: FloatArray, val gh: Int, val gw: Int, val dim: Int)

/** All exported models (mirrors runtime.Models). Image inputs are RGB 0..1 NCHW, resized here. */
class Models(
    val manifest: Manifest,
    private val sessions: SessionProvider,
    val clipBackground: Array<FloatArray>,
    val tokenizer: ClipTokenizer = ClipTokenizer(),
    scorerSession: OrtSession? = null,
) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val cache = HashMap<String, OrtSession>()
    val timings = Timings()
    private val scorer = scorerSession ?: env.createSession(scorerBytes(), CpuSessions.cpuOptions())

    fun backend(model: String) = sessions.backend(model)

    @Synchronized
    fun session(model: String, h: Int = 0, w: Int = 0): OrtSession =
        cache.getOrPut("$model:${h}x$w") { sessions.create(model, h, w) }

    /** Aspect-preserving input size: long side from the manifest, multiples of the stride. */
    fun inputSize(model: String, h: Int, w: Int): Pair<Int, Int> {
        val stride = manifest.stride(model)
        val s = manifest.longSide(model).toDouble() / maxOf(h, w)
        return maxOf(stride, Math.rint(h * s / stride).toInt() * stride) to
            maxOf(stride, Math.rint(w * s / stride).toInt() * stride)
    }

    private fun runFloat(session: OrtSession, input: String, data: FloatArray, shape: LongArray): FloatArray =
        OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape).use { t ->
            session.run(mapOf(input to t)).use { r -> (r.get(0) as OnnxTensor).floatBuffer.let { b -> FloatArray(b.remaining()).also { b.get(it) } } }
        }

    fun dinoPatches(frame: FloatImage): PatchFeats {
        val (h, w) = inputSize("dinov3_patches", frame.h, frame.w)
        val x = ImageOps.resizeLinear(frame.data, frame.w, 3, 0, 0, frame.w, frame.h, h, w, planar = true)
        val s = session("dinov3_patches", h, w)
        val out = timings.time("dino") { runFloat(s, "image", x, longArrayOf(1, 3, h.toLong(), w.toLong())) }
        val n = (h / 16) * (w / 16)
        return PatchFeats(out, h / 16, w / 16, out.size / n)
    }

    /** Segmenter on the [x0, x0+cw) × [y0, y0+ch) crop of the frame → probs [256*256]. */
    fun segment(frame: FloatImage, x0: Int, y0: Int, cw: Int, ch: Int): FloatArray {
        val x = ImageOps.resizeLinear(frame.data, frame.w, 3, x0, y0, cw, ch, 224, 224, planar = true)
        val s = session("segmenter")
        return timings.time("seg") { runFloat(s, "crop", x, longArrayOf(1, 3, 224, 224)) }
    }

    /** Dense CLIP features → [h/32, w/32, 512] (row-major), with its grid size. */
    fun denseClip(frame: FloatImage): Triple<FloatArray, Int, Int> {
        val (h, w) = inputSize("clip_dense", frame.h, frame.w)
        val x = ImageOps.resizeLinear(frame.data, frame.w, 3, 0, 0, frame.w, frame.h, h, w, planar = true)
        val s = session("clip_dense", h, w)
        val out = timings.time("clip") { runFloat(s, "image", x, longArrayOf(1, 3, h.toLong(), w.toLong())) }
        return Triple(out, h / 32, w / 32)
    }

    fun text(prompt: String): FloatArray {
        val tokens = tokenizer(prompt)
        val s = session("clip_text")
        return timings.time("text") {
            OnnxTensor.createTensor(env, LongBuffer.wrap(tokens), longArrayOf(1, tokens.size.toLong())).use { t ->
                s.run(mapOf("tokens" to t)).use { r -> (r.get(0) as OnnxTensor).floatBuffer.let { b -> FloatArray(b.remaining()).also { b.get(it) } } }
            }
        }
    }

    private var bankKey: Any? = null
    private var bankTensors: List<OnnxTensor> = emptyList()

    /**
     * Patch scores: top-k mean cosine to [fg] minus top-k mean cosine to [bg] (runtime._scores).
     * Bank tensors are reused while [bankKey] (the memory's version) is unchanged.
     */
    fun scores(feats: PatchFeats, fg: FloatArray, bg: FloatArray, k: Int, key: Any): FloatArray = timings.time("score") {
        val d = feats.dim.toLong()
        val nf = fg.size / feats.dim
        val nb = bg.size / feats.dim
        if (key != bankKey) {
            bankTensors.forEach { it.close() }
            bankTensors = listOf(
                OnnxTensor.createTensor(env, FloatBuffer.wrap(fg), longArrayOf(nf.toLong(), d)),
                OnnxTensor.createTensor(env, FloatBuffer.wrap(bg), longArrayOf(nb.toLong(), d)),
                OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(minOf(k, nf).toLong())), longArrayOf(1)),
                OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(minOf(k, nb).toLong())), longArrayOf(1)),
            )
            bankKey = key
        }
        val n = feats.gh * feats.gw
        OnnxTensor.createTensor(env, FloatBuffer.wrap(feats.data), longArrayOf(n.toLong(), d)).use { f ->
            val (tf, tb, kf, kb) = bankTensors
            scorer.run(mapOf("feats" to f, "fg" to tf, "bg" to tb, "k_fg" to kf, "k_bg" to kb)).use { r ->
                (r.get(0) as OnnxTensor).floatBuffer.let { b -> FloatArray(b.remaining()).also { b.get(it) } }
            }
        }
    }

    override fun close() {
        bankTensors.forEach { it.close() }
        cache.values.forEach { it.close() }
        scorer.close()
        sessions.close()
    }

    companion object {
        fun scorerBytes(): ByteArray =
            Models::class.java.classLoader!!.getResourceAsStream("scorer.onnx")!!.use { it.readBytes() }

        fun loadBackground(file: File): Array<FloatArray> = file.inputStream().use { FloatMatrix.read(it, 512) }

        /** CPU-only models from a mobile/models directory (fp32 vision models, INT8 text encoder). */
        fun cpu(dir: File, threads: Int = 0): Models {
            val manifest = Manifest.load(File(dir, "manifest.json"))
            val bg = File(dir, "clip_background.npy").let { if (it.exists()) it else File(dir, "clip_background.f32") }
            return Models(manifest, CpuSessions(dir, manifest, threads), loadBackground(bg))
        }
    }
}
