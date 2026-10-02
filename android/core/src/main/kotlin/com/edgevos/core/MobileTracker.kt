package com.edgevos.core

import com.edgevos.core.ImageOps.bbox
import com.edgevos.core.ImageOps.boxIou
import com.edgevos.core.ImageOps.boxMask
import com.edgevos.core.ImageOps.gridFraction
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/*
 * Kotlin port of mobile/runtime.py (Grounder, PatchMemory, MobileTracker), kept line for line.
 *
 * Numeric conventions follow the NumPy 2 reference: boxes are float32 and box arithmetic mixing in a
 * Python float stays float32 (NEP 50), and so do comparisons of float32 arrays against Python floats;
 * values the reference turns into Python floats (conf, peaks, diffs) are compared as doubles.
 */

class Detection(val box: Box, val score: Double, var mask: Mask? = null)

/** Dense MobileCLIP2 heatmap → candidate boxes (runtime.Grounder). */
class Grounder(private val m: Models, private val maxCandidates: Int = 3) {
    private val presence = m.manifest.presenceThreshold
    private val textCache = HashMap<String, FloatArray>()

    fun text(prompt: String): FloatArray =
        textCache.getOrPut(prompt) { m.text(m.manifest.promptTemplate.replace("{}", prompt)) }

    fun detect(frame: FloatImage, prompt: String): List<Detection> {
        val text = text(prompt)
        val (dense, gh, gw) = m.denseClip(frame)
        val dim = text.size
        val bgs = m.clipBackground
        val hm = FloatArray(gh * gw)
        var hmMax = Float.NEGATIVE_INFINITY
        for (i in 0 until gh * gw) {
            val o = i * dim
            var t = 0f
            for (k in 0 until dim) t += dense[o + k] * text[k]
            var b = Float.NEGATIVE_INFINITY
            for (bg in bgs) {
                var s = 0f
                for (k in 0 until dim) s += dense[o + k] * bg[k]
                if (s > b) b = s
            }
            hm[i] = t - b
            if (hm[i] > hmMax) hmMax = hm[i]
        }
        if (hmMax < presence) return emptyList()
        val up = ImageOps.resizeLinear(hm, gh, gw, frame.h, frame.w)
        var lo = Float.POSITIVE_INFINITY
        var hi = Float.NEGATIVE_INFINITY
        for (v in up) { if (v < lo) lo = v; if (v > hi) hi = v }
        val thr = (lo.toDouble() + 0.6 * (hi.toDouble() - lo.toDouble())).toFloat()
        val (n, labels) = ImageOps.connectedComponents(BooleanArray(up.size) { up[it] >= thr }, frame.h, frame.w)
        val peaks = FloatArray(n) { Float.NEGATIVE_INFINITY }
        for (i in up.indices) if (labels[i] > 0 && up[i] > peaks[labels[i]]) peaks[labels[i]] = up[i]
        val dets = ArrayList<Detection>()
        for (k in 1 until n) {
            val peak = peaks[k].toDouble()
            if (peak >= presence.toDouble()) {
                val comp = Mask(frame.h, frame.w, BooleanArray(up.size) { labels[it] == k })
                dets += Detection(bbox(comp), peak)
            }
        }
        dets.sortByDescending { it.score }       // stable, like list.sort(reverse=True)
        return dets.take(maxCandidates)
    }
}

/** Anchor slot + FIFO of recent keyframes of fg/bg patch features (runtime.PatchMemory). */
class PatchMemory(private val recent: Int, private val maxFg: Int, private val maxBg: Int, private val dim: Int, seed: Long = 0) {
    class Slot(val fg: FloatArray, val bg: FloatArray)

    var anchor: Slot? = null
        private set
    private val recentSlots = ArrayDeque<Slot>()
    private val rng = NumpyRandom(seed)
    /** Bumped on every write; keys the cached bank tensors. */
    var version = 0
        private set
    private var banksCache: Pair<FloatArray, FloatArray>? = null
    private var banksVersion = -1

    private fun cap(x: FloatArray, n: Int): FloatArray {
        val rows = x.size / dim
        if (rows <= n) return x
        val perm = rng.permutation(rows)
        val out = FloatArray(n * dim)
        for (i in 0 until n) System.arraycopy(x, perm[i] * dim, out, i * dim, dim)
        return out
    }

    fun setAnchor(fg: FloatArray, bg: FloatArray) {
        anchor = Slot(cap(fg, maxFg), cap(bg, maxBg))
        recentSlots.clear()
        version++
    }

    fun add(fg: FloatArray, bg: FloatArray) {
        val slot = Slot(cap(fg, maxFg / 2), cap(bg, maxBg / 2))
        if (recentSlots.size == recent) recentSlots.removeFirst()
        recentSlots.addLast(slot)
        version++
    }

    fun banks(): Pair<FloatArray, FloatArray> {
        if (banksVersion != version) {
            val slots = listOf(anchor!!) + recentSlots
            fun cat(sel: (Slot) -> FloatArray): FloatArray {
                val out = FloatArray(slots.sumOf { sel(it).size })
                var o = 0
                for (s in slots) { val a = sel(s); a.copyInto(out, o); o += a.size }
                return out
            }
            banksCache = cat { it.fg } to cat { it.bg }
            banksVersion = version
        }
        return banksCache!!
    }
}

/** Text-prompted tracker (runtime.MobileTracker): `init(frame, prompt)`, then `step(frame)` per frame. */
class MobileTracker(val models: Models, cfgOverride: Map<String, Any?> = emptyMap()) {
    val cfg = TrackerConfig(models.manifest.trackerCfg + cfgOverride)
    val grounder = Grounder(models)

    var prompt: String? = null
        private set
    private lateinit var memory: PatchMemory
    var frameT = 0
        private set
    private var box: Box? = null
    private var areaEma: Double? = null
    private var lowConfCount = 0
    private var lastRegroundT = 0
    private var absent = false
    private var pendingBox: Box? = null
    private var prevThumb: FloatArray? = null
    private var prevFeats: PatchFeats? = null
    private val grayDiffs = ArrayDeque<Double>()
    var lastConf = 0.0
        private set
    /** "init", "absent", "reground:<reason>" or null on a plain tracked frame. */
    var lastEvent: String? = null
        private set

    init { reset() }

    fun reset() {
        prompt = null
        memory = PatchMemory(cfg.recentKeyframes, cfg.maxFgBank, cfg.maxBgBank, DIM)
        frameT = 0
        box = null
        areaEma = null
        lowConfCount = 0
        lastRegroundT = 0
        absent = false
        pendingBox = null
        prevThumb = null
        prevFeats = null
        grayDiffs.clear()
        lastConf = 0.0
        lastEvent = null
    }

    // Public API -----------------------------------------------------------------------------

    /** frame: H×W×3 RGB in [0, 1]. Returns the H×W mask, or null when the object is absent. */
    fun init(frame: FloatImage, prompt: String): Mask? {
        models.timings.reset()
        reset()
        this.prompt = prompt
        val ff = models.dinoPatches(frame)
        sceneCut(frame, ff)
        val dets = grounder.detect(frame, prompt)
        if (dets.isEmpty()) {
            absent = true; lastEvent = "absent"
            return null
        }
        val mask = detectionMask(frame, dets[0])
        commit(ff, mask, anchor = true)
        lastEvent = "init"
        return mask
    }

    fun step(frame: FloatImage): Mask? {
        models.timings.reset()
        val c = cfg
        frameT += 1
        lastEvent = null
        val ff = models.dinoPatches(frame)
        val cut = sceneCut(frame, ff)

        if (memory.anchor == null || absent) {
            if (cut || frameT - lastRegroundT >= c.absentRetry) return reground(frame, ff, if (cut) "scene_cut" else "absent")
            return null
        }
        if (cut) return reground(frame, ff, "scene_cut")

        val (mask, conf) = propagate(frame, ff)
        lastConf = conf
        lowConfCount = if (mask == null || conf < c.lowConf) lowConfCount + 1 else 0
        val cooled = frameT - lastRegroundT >= c.regroundCooldown
        if (lowConfCount >= c.lowConfFrames && cooled) return reground(frame, ff, "low_confidence", fallback = mask)
        if (c.regroundInterval != 0 && frameT - lastRegroundT >= c.regroundInterval)
            return reground(frame, ff, "periodic", fallback = mask)
        if (mask != null) {
            val ratio = areaEma?.let { mask.count() / it }
            val areaOk = ratio == null || (ratio >= 0.5 && ratio <= 2.0)
            val write = frameT % c.keyframeInterval == 0 && conf >= c.keyframeMinConf && areaOk
            commit(ff, mask, keyframe = write)
        }
        return mask
    }

    // Propagation ----------------------------------------------------------------------------

    private fun scores(ff: PatchFeats): FloatArray {
        val (fg, bg) = memory.banks()
        return models.scores(ff, fg, bg, cfg.knn, memory to memory.version)
    }

    private fun propagate(frame: FloatImage, ff: PatchFeats): Pair<Mask?, Double> {
        val c = cfg
        val H = frame.h
        val W = frame.w
        val gh = ff.gh
        val gw = ff.gw
        val score = scores(ff)
        val (x1, y1, x2, y2) = box!!
        val m = c.searchMargin * max(x2 - x1, y2 - y1)
        val window = BooleanArray(gh * gw)
        val wy0 = max(0, ((y1 - m) / H * gh).toInt()); val wy1 = min(gh, ceil((y2 + m) / H * gh).toInt())
        val wx0 = max(0, ((x1 - m) / W * gw).toInt()); val wx1 = min(gw, ceil((x2 + m) / W * gw).toInt())
        for (y in wy0 until wy1) for (x in wx0 until wx1) window[y * gw + x] = true
        val (n, labels) = ImageOps.connectedComponents(BooleanArray(gh * gw) { score[it] > 0 && window[it] }, gh, gw)
        if (n <= 1) {
            var best = Float.NEGATIVE_INFINITY
            for (i in score.indices) if (window[i] && score[i] > best) best = score[i]
            return null to best.toDouble()
        }

        val prev = BooleanArray(gh * gw)
        val py0 = (y1 / H * gh).toInt(); val py1 = min(gh, ceil(y2 / H * gh).toInt())
        val px0 = (x1 / W * gw).toInt(); val px1 = min(gw, ceil(x2 / W * gw).toInt())
        for (y in py0 until py1) for (x in px0 until px1) prev[y * gw + x] = true
        val overlap = IntArray(n)
        val posSum = DoubleArray(n)
        for (i in labels.indices) {
            val l = labels[i]
            if (l == 0) continue
            if (prev[i]) overlap[l]++
            posSum[l] += max(score[i], 0f).toDouble()
        }
        var best = 1
        for (k in 2 until n) if (overlap[k] > overlap[best] || (overlap[k] == overlap[best] && posSum[k] > posSum[best])) best = k
        val comp = BooleanArray(gh * gw) { labels[it] == best }
        var sum = 0.0
        var cnt = 0
        var gx1 = Int.MAX_VALUE; var gy1 = Int.MAX_VALUE; var gx2 = -1; var gy2 = -1
        for (i in comp.indices) if (comp[i]) {
            sum += score[i]; cnt++
            val y = i / gw; val x = i % gw
            gx1 = min(gx1, x); gx2 = max(gx2, x); gy1 = min(gy1, y); gy2 = max(gy2, y)
        }
        val conf = (sum / cnt).toFloat().toDouble()

        val cw = W.toDouble() / gw
        val ch = H.toDouble() / gh
        val coarse = floatArrayOf((gx1 * cw).toFloat(), (gy1 * ch).toFloat(), ((gx2 + 1) * cw).toFloat(), ((gy2 + 1) * ch).toFloat())
        val dil = ImageOps.dilate3x3(comp, gh, gw)
        val gateUp = ImageOps.resizeLinear(FloatArray(gh * gw) { if (dil[it]) 1f else 0f }, gh, gw, H, W)
        val gate = BooleanArray(H * W) { gateUp[it] > 0.5f }

        val probs = maskProbs(frame, coarse)
        var mask = Mask(H, W, BooleanArray(H * W) { probs[it] > c.maskThreshold && gate[it] })
        if (mask.count() < 0.15 * gate.count { it }) {
            val up = ImageOps.resizeLinear(score, gh, gw, H, W)
            mask = Mask(H, W, BooleanArray(H * W) { up[it] > 0f && gate[it] })
        }
        if (!mask.any()) return null to conf
        return mask to conf
    }

    // Re-grounding ---------------------------------------------------------------------------

    private fun reground(frame: FloatImage, ff: PatchFeats, reason: String, fallback: Mask? = null): Mask? {
        lastRegroundT = frameT
        val det = select(frame, ff)
        if (det == null) {
            if (fallback != null && reason == "periodic") {
                commit(ff, fallback)
                return fallback
            }
            absent = true; lowConfCount = 0; pendingBox = null
            lastEvent = "absent"
            return null
        }
        if (absent || memory.anchor == null) {             // confirm re-acquisition
            val pending = pendingBox
            pendingBox = det.box
            if (pending == null || boxIou(pending, det.box) < 0.3f) {
                absent = true; lastEvent = "absent"
                return null
            }
        }
        pendingBox = null
        val mask = detectionMask(frame, det)
        val first = memory.anchor == null
        commit(ff, mask, anchor = first, keyframe = !first)
        absent = false; lowConfCount = 0
        lastEvent = "reground:$reason"
        return mask
    }

    private fun select(frame: FloatImage, ff: PatchFeats): Detection? {
        val dets = grounder.detect(frame, prompt!!)
        if (dets.isEmpty()) return null
        if (memory.anchor == null) return dets[0]
        val score = scores(ff)
        var best: Detection? = null
        var bestVal = -1e9
        for (d in dets) {
            val mask = detectionMask(frame, d)
            d.mask = mask
            val frac = gridFraction(mask, ff.gh, ff.gw)
            var s = 0.0
            var n = 0
            for (i in frac.indices) if (frac[i] > 0.5f) { s += score[i]; n++ }
            val appearance = if (n > 0) (s / n).toFloat().toDouble() else -1.0
            if (appearance < cfg.minDetAppearance && d.score < 0.6) continue
            if (d.score + 4.0 * appearance > bestVal) { best = d; bestVal = d.score + 4.0 * appearance }
        }
        return best
    }

    private fun detectionMask(frame: FloatImage, det: Detection): Mask {
        det.mask?.let { if (it.any()) return it }
        val H = frame.h
        val W = frame.w
        var box = det.box
        var mask: Mask? = null
        repeat(cfg.boxRefineIters) {
            val probs = maskProbs(frame, box)
            val bm = boxMask(box, H, W, pad = 0.15f)
            val m = Mask(H, W, BooleanArray(H * W) { probs[it] > cfg.maskThreshold && bm.data[it] })
            if (!m.any()) return mask ?: boxMask(det.box, H, W)
            mask = m
            box = bbox(m)
        }
        return mask ?: boxMask(det.box, H, W)
    }

    // State ----------------------------------------------------------------------------------

    private fun commit(ff: PatchFeats, mask: Mask, anchor: Boolean = false, keyframe: Boolean = false) {
        if (!mask.any()) return
        if (anchor || keyframe) {
            val frac = gridFraction(mask, ff.gh, ff.gw)
            var fg = rows(ff, frac) { it > 0.5f }
            if (fg.isEmpty()) {
                var arg = 0
                for (i in frac.indices) if (frac[i] > frac[arg]) arg = i
                fg = ff.data.copyOfRange(arg * ff.dim, (arg + 1) * ff.dim)
            }
            val bg = rows(ff, frac) { it < 0.05f }
            if (anchor) memory.setAnchor(fg, bg) else memory.add(fg, bg)
        }
        box = bbox(mask)
        val area = mask.count().toDouble()
        areaEma = areaEma?.let { 0.8 * it + 0.2 * area } ?: area
    }

    private inline fun rows(ff: PatchFeats, frac: FloatArray, keep: (Float) -> Boolean): FloatArray {
        val idx = frac.indices.filter { keep(frac[it]) }
        val out = FloatArray(idx.size * ff.dim)
        idx.forEachIndexed { j, i -> System.arraycopy(ff.data, i * ff.dim, out, j * ff.dim, ff.dim) }
        return out
    }

    /** Segmenter probabilities on the box +15% (runtime._mask_probs), pasted into an H×W map. */
    private fun maskProbs(frame: FloatImage, box: Box): FloatArray {
        val H = frame.h
        val W = frame.w
        val (x1, y1, x2, y2) = box
        val pad = max(max(x2 - x1, y2 - y1), 1f) * 0.15f
        val ex1 = max(0, (x1 - pad).toInt()); val ey1 = max(0, (y1 - pad).toInt())
        val ex2 = min(W, ceil(x2 + pad).toInt()); val ey2 = min(H, ceil(y2 + pad).toInt())
        val full = FloatArray(H * W)
        if (ex2 - ex1 < 2 || ey2 - ey1 < 2) return full
        val probs = models.segment(frame, ex1, ey1, ex2 - ex1, ey2 - ey1)
        val cw = ex2 - ex1
        val crop = ImageOps.resizeLinear(probs, 256, 256, ey2 - ey1, cw)
        for (y in 0 until ey2 - ey1) System.arraycopy(crop, y * cw, full, (ey1 + y) * W + ex1, cw)
        return full
    }

    private fun sceneCut(frame: FloatImage, ff: PatchFeats): Boolean {
        val c = cfg
        val gray = FloatArray(frame.h * frame.w)
        val d = frame.data
        for (i in gray.indices) gray[i] = (d[3 * i] + d[3 * i + 1] + d[3 * i + 2]) / 3f
        val thumb = ImageOps.resizeArea(gray, frame.h, frame.w, 36, 64)
        for (i in thumb.indices) thumb[i] *= 255f
        var cut = false
        val prev = prevThumb
        if (prev != null) {
            var s = 0.0
            for (i in thumb.indices) s += abs(thumb[i] - prev[i])
            val diff = (s / thumb.size).toFloat().toDouble()
            val median = if (grayDiffs.isEmpty()) 0.0 else grayDiffs.sorted().let { v ->
                if (v.size % 2 == 1) v[v.size / 2] else (v[v.size / 2 - 1] + v[v.size / 2]) / 2
            }
            cut = diff > c.cutGrayDiff && diff > c.cutGrayRatio * median
            val pf = prevFeats
            if (!cut && pf != null && pf.data.size == ff.data.size) {
                var low = 0
                val n = ff.gh * ff.gw
                for (i in 0 until n) {
                    var dot = 0f
                    val o = i * ff.dim
                    for (k in 0 until ff.dim) dot += ff.data[o + k] * pf.data[o + k]
                    if (dot < 0.5f) low++
                }
                cut = low.toDouble() / n > c.cutPatchFrac
            }
            if (!cut) {
                if (grayDiffs.size == 30) grayDiffs.removeFirst()
                grayDiffs.addLast(diff)
            }
        }
        prevThumb = thumb
        prevFeats = ff
        return cut
    }

    companion object {
        const val DIM = 384
    }
}

/** H×W×3 RGB bytes → float image in [0, 1] (runtime.MobileTracker._as_float). */
fun FloatImage.Companion.fromRgb(rgb: ByteArray, h: Int, w: Int): FloatImage {
    val out = FloatArray(h * w * 3)
    for (i in out.indices) out[i] = (rgb[i].toInt() and 0xff) / 255f
    return FloatImage(h, w, 3, out)
}
