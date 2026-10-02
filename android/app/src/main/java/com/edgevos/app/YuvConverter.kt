package com.edgevos.app

import androidx.camera.core.ImageProxy
import com.edgevos.core.FloatImage
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * YUV_420_888 camera frame → upright RGB float image in [0, 1], in one pass: crops to the image's
 * cropRect, box-downscales by an integer factor (long side ≈ [targetLongSide]) and rotates by
 * `imageInfo.rotationDegrees`. Uses full-range BT.601 (JFIF), the camera's YUV encoding.
 */
class YuvConverter(private val targetLongSide: Int) {
    private var y = ByteArray(0)
    private var u = ByteArray(0)
    private var v = ByteArray(0)

    fun convert(image: ImageProxy): FloatImage {
        val crop = image.cropRect
        val planes = image.planes
        y = copy(planes[0].buffer, y); u = copy(planes[1].buffer, u); v = copy(planes[2].buffer, v)
        return convert(
            Planes(y, planes[0].rowStride, planes[0].pixelStride, u, planes[1].rowStride, planes[1].pixelStride,
                v, planes[2].rowStride, planes[2].pixelStride),
            crop.left, crop.top, crop.width(), crop.height(), image.imageInfo.rotationDegrees,
        )
    }

    class Planes(
        val y: ByteArray, val yRow: Int, val yPix: Int,
        val u: ByteArray, val uRow: Int, val uPix: Int,
        val v: ByteArray, val vRow: Int, val vPix: Int,
    )

    fun convert(p: Planes, left: Int, top: Int, width: Int, height: Int, rotation: Int): FloatImage {
        val s = max(1, (max(width, height).toDouble() / targetLongSide).roundToInt())
        val sw = width / s
        val sh = height / s
        val rotated = rotation == 90 || rotation == 270
        val ow = if (rotated) sh else sw
        val oh = if (rotated) sw else sh
        val out = FloatArray(ow * oh * 3)
        val inv = 1f / (s * s)
        val y = p.y
        for (oy in 0 until sh) {
            val y0 = top + oy * s
            for (ox in 0 until sw) {
                val x0 = left + ox * s
                var ys = 0
                for (dy in 0 until s) {
                    val r = (y0 + dy) * p.yRow
                    for (dx in 0 until s) ys += y[r + (x0 + dx) * p.yPix].toInt() and 0xff
                }
                val cx = (x0 + s / 2) shr 1
                val cy = (y0 + s / 2) shr 1
                val yy = ys * inv
                val uu = (p.u[cy * p.uRow + cx * p.uPix].toInt() and 0xff) - 128f
                val vv = (p.v[cy * p.vRow + cx * p.vPix].toInt() and 0xff) - 128f
                // Destination pixel in the upright image.
                val o = 3 * when (rotation) {
                    90 -> ox * ow + (sh - 1 - oy)
                    180 -> (sh - 1 - oy) * ow + (sw - 1 - ox)
                    270 -> (sw - 1 - ox) * ow + oy
                    else -> oy * ow + ox
                }
                out[o] = clamp(yy + 1.402f * vv)
                out[o + 1] = clamp(yy - 0.344136f * uu - 0.714136f * vv)
                out[o + 2] = clamp(yy + 1.772f * uu)
            }
        }
        return FloatImage(oh, ow, 3, out)
    }

    private fun clamp(x: Float) = (if (x < 0f) 0f else if (x > 255f) 255f else x) / 255f

    private fun copy(buf: java.nio.ByteBuffer, dst: ByteArray): ByteArray {
        buf.rewind()
        val out = if (dst.size == buf.remaining()) dst else ByteArray(buf.remaining())
        buf.get(out)
        return out
    }
}
