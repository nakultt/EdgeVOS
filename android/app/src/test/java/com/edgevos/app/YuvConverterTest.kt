package com.edgevos.app

import org.junit.Assert.assertEquals
import org.junit.Test

/** Rotation, crop and downscale of the YUV → RGB path, on synthetic frames. */
class YuvConverterTest {
    /** A W×H frame whose luma encodes the pixel position and whose chroma is neutral (grey RGB = Y). */
    private fun frame(w: Int, h: Int, uvPix: Int, luma: (Int, Int) -> Int): YuvConverter.Planes {
        val y = ByteArray(w * h) { luma(it % w, it / w).toByte() }
        val cw = w / 2
        val ch = h / 2
        // Semi-planar layout (pixelStride 2) like most camera HALs, or planar (1).
        val uv = ByteArray(cw * ch * uvPix) { 128.toByte() }
        return YuvConverter.Planes(y, w, 1, uv, cw * uvPix, uvPix, uv, cw * uvPix, uvPix)
    }

    @Test
    fun rotatesUpright() {
        // 8×4 landscape sensor frame, top-left pixel marked.
        val luma = { x: Int, yy: Int -> if (x == 0 && yy == 0) 200 else 10 }
        val conv = YuvConverter(8)
        for ((rot, expected) in listOf(0 to (0 to 0), 90 to (3 to 0), 180 to (7 to 3), 270 to (0 to 7))) {
            val img = conv.convert(frame(8, 4, 2, luma), 0, 0, 8, 4, rot)
            val rotated = rot == 90 || rot == 270
            assertEquals(if (rotated) 4 else 8, img.w)
            assertEquals(if (rotated) 8 else 4, img.h)
            val (ex, ey) = expected
            assertEquals("rot $rot", 200f / 255f, img.data[(ey * img.w + ex) * 3], 1e-6f)
            assertEquals(1, img.data.count { it > 0.5f } / 3)
        }
    }

    @Test
    fun downscalesByBoxAverageAndCrops() {
        // 1280×720 → long side 640: factor 2, each output pixel averages a 2×2 block.
        val luma = { x: Int, yy: Int -> (x + yy) % 200 }
        val img = YuvConverter(640).convert(frame(1280, 720, 1, luma), 0, 0, 1280, 720, 90)
        assertEquals(360, img.w)
        assertEquals(640, img.h)
        // Upright (r, c) comes from sensor (x = r, y = 719 - c) in output-pixel units.
        for ((r, c) in listOf(0 to 0, 10 to 5, 639 to 359, 300 to 200)) {
            val sx = r * 2
            val sy = (359 - c) * 2
            val avg = (luma(sx, sy) + luma(sx + 1, sy) + luma(sx, sy + 1) + luma(sx + 1, sy + 1)) / 4f
            assertEquals("($r,$c)", avg / 255f, img.data[(r * img.w + c) * 3 + 1], 1e-5f)
        }
        // A crop rect offsets the source window.
        val cropped = YuvConverter(100).convert(frame(1280, 720, 1, luma), 640, 360, 200, 100, 0)
        assertEquals(100, cropped.w)
        assertEquals(50, cropped.h)
        val avg = (luma(640, 360) + luma(641, 360) + luma(640, 361) + luma(641, 361)) / 4f
        assertEquals(avg / 255f, cropped.data[0], 1e-5f)
    }

    @Test
    fun convertsColour() {
        // Pure red in full-range BT.601: Y=76, U=85, V=255.
        val p = YuvConverter.Planes(ByteArray(4) { 76 }, 2, 1, ByteArray(1) { 85 }, 1, 1, ByteArray(1) { 255.toByte() }, 1, 1)
        val img = YuvConverter(2).convert(p, 0, 0, 2, 2, 0)
        assertEquals(1f, img.data[0], 0.01f)
        assertEquals(0f, img.data[1], 0.01f)
        assertEquals(0f, img.data[2], 0.01f)
    }
}
