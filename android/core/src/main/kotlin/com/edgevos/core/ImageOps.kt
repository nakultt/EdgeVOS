package com.edgevos.core

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/** H×W×C float image in row-major HWC order (a numpy array). RGB frames hold values in [0, 1]. */
class FloatImage(val h: Int, val w: Int, val c: Int, val data: FloatArray = FloatArray(h * w * c)) {
    init { require(data.size == h * w * c) }

    companion object
}

/** H×W boolean mask, row-major. */
class Mask(val h: Int, val w: Int, val data: BooleanArray = BooleanArray(h * w)) {
    fun count(): Int { var n = 0; for (v in data) if (v) n++; return n }
    fun any(): Boolean { for (v in data) if (v) return true; return false }
    operator fun get(y: Int, x: Int) = data[y * w + x]
}

/** Box as float32 [x1, y1, x2, y2] (numpy float32 semantics, like runtime.py). */
typealias Box = FloatArray

/**
 * Ports of the small numpy / OpenCV helpers in mobile/runtime.py. Resizes reproduce OpenCV's
 * float `INTER_LINEAR` (half-pixel centres, clamped edges, horizontal-then-vertical pass) and
 * `INTER_AREA` (fractional box weights) so the tracker sees the same numbers as the Python reference.
 */
object ImageOps {

    /** OpenCV's per-axis linear interpolation table (cv::resize, INTER_LINEAR). */
    private class LinearTab(src: Int, dst: Int) {
        val i0 = IntArray(dst)
        val i1 = IntArray(dst)
        val a0 = FloatArray(dst)
        val a1 = FloatArray(dst)

        init {
            val scale = 1.0 / (dst.toDouble() / src)
            for (d in 0 until dst) {
                var f = ((d + 0.5) * scale - 0.5).toFloat()
                var s = floor(f).toInt()
                f -= s
                if (s < 0) { f = 0f; s = 0 }
                if (s >= src - 1) { f = 0f; s = src - 1 }
                i0[d] = s
                i1[d] = min(s + 1, src - 1)
                a0[d] = 1f - f
                a1[d] = f
            }
        }
    }

    /**
     * Bilinear resize of the [x0, x0+sw) × [y0, y0+sh) region of an HWC image (row width [srcW],
     * [c] channels) to dh × dw. Output is HWC, or planar CHW when [planar] (an NCHW model input).
     */
    fun resizeLinear(
        src: FloatArray, srcW: Int, c: Int, x0: Int, y0: Int, sw: Int, sh: Int,
        dh: Int, dw: Int, out: FloatArray = FloatArray(dh * dw * c), planar: Boolean = false,
    ): FloatArray {
        val tx = LinearTab(sw, dw)
        val ty = LinearTab(sh, dh)
        val row0 = FloatArray(dw * c)
        val row1 = FloatArray(dw * c)
        var cached0 = -1
        var cached1 = -1
        fun hpass(sy: Int, row: FloatArray) {
            val base = ((y0 + sy) * srcW + x0) * c
            for (dx in 0 until dw) {
                val p0 = base + tx.i0[dx] * c
                val p1 = base + tx.i1[dx] * c
                val a0 = tx.a0[dx]
                val a1 = tx.a1[dx]
                for (k in 0 until c) row[dx * c + k] = src[p0 + k] * a0 + src[p1 + k] * a1
            }
        }
        val plane = dh * dw
        for (dy in 0 until dh) {
            val sy0 = ty.i0[dy]
            val sy1 = ty.i1[dy]
            // Rows advance monotonically; reuse the previous horizontal pass when possible.
            if (sy0 != cached0) {
                if (sy0 == cached1) row1.copyInto(row0) else hpass(sy0, row0)
                cached0 = sy0
            }
            if (sy1 != cached1 || cached1 == -1) { hpass(sy1, row1); cached1 = sy1 }
            val b0 = ty.a0[dy]
            val b1 = ty.a1[dy]
            if (planar) {
                for (dx in 0 until dw) for (k in 0 until c)
                    out[k * plane + dy * dw + dx] = row0[dx * c + k] * b0 + row1[dx * c + k] * b1
            } else {
                val o = dy * dw * c
                for (i in 0 until dw * c) out[o + i] = row0[i] * b0 + row1[i] * b1
            }
        }
        return out
    }

    /** Bilinear resize of a whole single-channel H×W map. */
    fun resizeLinear(src: FloatArray, h: Int, w: Int, dh: Int, dw: Int): FloatArray =
        resizeLinear(src, w, 1, 0, 0, w, h, dh, dw)

    /** OpenCV's INTER_AREA weight table: (dst index, src index, weight) triples. */
    private fun areaTab(src: Int, dst: Int): Triple<IntArray, IntArray, FloatArray> {
        val scale = src.toDouble() / dst
        val di = ArrayList<Int>()
        val si = ArrayList<Int>()
        val wt = ArrayList<Float>()
        for (dx in 0 until dst) {
            val fsx1 = dx * scale
            val fsx2 = fsx1 + scale
            val cellWidth = min(scale, src - fsx1)
            var sx1 = ceil(fsx1).toInt()
            var sx2 = floor(fsx2).toInt()
            sx2 = min(sx2, src - 1)
            sx1 = min(sx1, sx2)
            if (sx1 - fsx1 > 1e-3) { di += dx; si += sx1 - 1; wt += ((sx1 - fsx1) / cellWidth).toFloat() }
            for (sx in sx1 until sx2) { di += dx; si += sx; wt += (1.0 / cellWidth).toFloat() }
            if (fsx2 - sx2 > 1e-3) {
                di += dx; si += sx2; wt += (min(min(fsx2 - sx2, 1.0), cellWidth) / cellWidth).toFloat()
            }
        }
        return Triple(di.toIntArray(), si.toIntArray(), wt.toFloatArray())
    }

    /** INTER_AREA downscale of a single-channel H×W map (cv2.resize(..., INTER_AREA)). */
    fun resizeArea(src: FloatArray, h: Int, w: Int, dh: Int, dw: Int): FloatArray {
        val (xd, xs, xw) = areaTab(w, dw)
        val (yd, ys, yw) = areaTab(h, dh)
        val rows = Array(h) { y ->
            val r = FloatArray(dw)
            val base = y * w
            for (k in xd.indices) r[xd[k]] += src[base + xs[k]] * xw[k]
            r
        }
        val out = FloatArray(dh * dw)
        for (k in yd.indices) {
            val r = rows[ys[k]]
            val o = yd[k] * dw
            val a = yw[k]
            for (x in 0 until dw) out[o + x] += r[x] * a
        }
        return out
    }

    /** cv2.dilate with a 3×3 square kernel on a binary grid. */
    fun dilate3x3(src: BooleanArray, h: Int, w: Int): BooleanArray {
        val out = BooleanArray(h * w)
        for (y in 0 until h) for (x in 0 until w) {
            if (!src[y * w + x]) continue
            for (yy in max(0, y - 1)..min(h - 1, y + 1)) for (xx in max(0, x - 1)..min(w - 1, x + 1)) out[yy * w + xx] = true
        }
        return out
    }

    /**
     * 8-connected components (cv2.connectedComponents(..., connectivity=8)).
     * Returns (n, labels): labels are 1..n-1 in raster order of each component's first pixel, 0 is background.
     */
    fun connectedComponents(bin: BooleanArray, h: Int, w: Int): Pair<Int, IntArray> {
        val labels = IntArray(h * w)
        val stack = IntArray(h * w)
        var n = 1
        for (start in 0 until h * w) {
            if (!bin[start] || labels[start] != 0) continue
            var sp = 0
            stack[sp++] = start
            labels[start] = n
            while (sp > 0) {
                val p = stack[--sp]
                val y = p / w
                val x = p - y * w
                for (yy in max(0, y - 1)..min(h - 1, y + 1)) for (xx in max(0, x - 1)..min(w - 1, x + 1)) {
                    val q = yy * w + xx
                    if (bin[q] && labels[q] == 0) { labels[q] = n; stack[sp++] = q }
                }
            }
            n++
        }
        return n to labels
    }

    /** Bins of adaptive average pooling (runtime.pool_matrix): output i averages [start, end). */
    fun poolBins(inSize: Int, outSize: Int): Array<IntArray> =
        Array(outSize) { i -> intArrayOf((i * inSize) / outSize, -Math.floorDiv(-(i + 1) * inSize, outSize)) }

    /** Fraction of each gh×gw patch cell covered by the mask (runtime.grid_fraction). */
    fun gridFraction(mask: Mask, gh: Int, gw: Int): FloatArray {
        val rb = poolBins(mask.h, gh)
        val cb = poolBins(mask.w, gw)
        // Per-row prefix sums make each cell count O(rows).
        val prefix = IntArray((mask.w + 1) * mask.h)
        for (y in 0 until mask.h) {
            val o = y * (mask.w + 1)
            for (x in 0 until mask.w) prefix[o + x + 1] = prefix[o + x] + if (mask.data[y * mask.w + x]) 1 else 0
        }
        val out = FloatArray(gh * gw)
        for (i in 0 until gh) for (j in 0 until gw) {
            var cnt = 0L
            val (c0, c1) = cb[j].let { it[0] to it[1] }
            for (y in rb[i][0] until rb[i][1]) cnt += prefix[y * (mask.w + 1) + c1] - prefix[y * (mask.w + 1) + c0]
            out[i * gw + j] = (cnt.toDouble() / ((rb[i][1] - rb[i][0]).toDouble() * (c1 - c0))).toFloat()
        }
        return out
    }

    fun bbox(mask: Mask): Box {
        var x1 = Int.MAX_VALUE; var y1 = Int.MAX_VALUE; var x2 = -1; var y2 = -1
        for (y in 0 until mask.h) {
            val o = y * mask.w
            for (x in 0 until mask.w) if (mask.data[o + x]) {
                if (x < x1) x1 = x
                if (x > x2) x2 = x
                if (y < y1) y1 = y
                y2 = y
            }
        }
        return floatArrayOf(x1.toFloat(), y1.toFloat(), (x2 + 1).toFloat(), (y2 + 1).toFloat())
    }

    /** runtime.box_mask: the box (grown by [pad] × its longer side) as an H×W mask. */
    fun boxMask(box: Box, h: Int, w: Int, pad: Float = 0f): Mask {
        val (x1, y1, x2, y2) = box
        val p = max(x2 - x1, y2 - y1) * pad
        val m = Mask(h, w)
        val ya = max(0, (y1 - p).toInt()); val yb = min(h, ceil(y2 + p).toInt())
        val xa = max(0, (x1 - p).toInt()); val xb = min(w, ceil(x2 + p).toInt())
        for (y in ya until yb) for (x in xa until xb) m.data[y * w + x] = true
        return m
    }

    fun boxIou(a: Box, b: Box): Float {
        val ix = max(0f, min(a[2], b[2]) - max(a[0], b[0]))
        val iy = max(0f, min(a[3], b[3]) - max(a[1], b[1]))
        val inter = ix * iy
        return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-6f)
    }
}
