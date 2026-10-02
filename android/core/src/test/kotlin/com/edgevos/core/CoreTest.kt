package com.edgevos.core

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Kotlin helpers vs golden values from the Python reference (android/tools/make_fixtures.py). */
class CoreTest {
    private fun res(name: String) = javaClass.classLoader.getResourceAsStream("fixtures/$name")!!.use { it.readBytes() }
    private fun f32(name: String): FloatArray =
        ByteBuffer.wrap(res(name)).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().let { b -> FloatArray(b.remaining()).also { b.get(it) } }

    @Suppress("UNCHECKED_CAST")
    private val fx = Json.parse(String(res("fixtures.json"))) as Map<String, Any?>

    private fun maxDiff(a: FloatArray, b: FloatArray): Float {
        assertEquals(a.size, b.size)
        var m = 0f
        for (i in a.indices) m = maxOf(m, abs(a[i] - b[i]))
        return m
    }

    @Test
    fun tokenizerMatchesPython() {
        val tok = ClipTokenizer()
        @Suppress("UNCHECKED_CAST")
        val tokens = fx["tokens"] as Map<String, List<Double>>
        for ((prompt, ids) in tokens) {
            assertContentEquals(ids.map { it.toLong() }.toLongArray(), tok(prompt), "prompt '$prompt'")
        }
    }

    @Test
    fun numpyRandomMatchesPython() {
        @Suppress("UNCHECKED_CAST")
        val rng = fx["rng"] as Map<String, List<Any>>
        val state = NumpyRandom.seedSequenceState(0, 8).map { it.toLong() and 0xffffffffL }
        assertEquals(rng["seed_state"]!!.map { (it as Double).toLong() }, state)
        val r0 = NumpyRandom(0)
        // JSON doubles lose precision on 64-bit values, so compare through their decimal strings.
        val raw = String(res("fixtures.json")).substringAfter("\"raw\": [").substringBefore("]").split(",").map { it.trim() }
        assertEquals(raw, List(4) { java.lang.Long.toUnsignedString(r0.nextLong()) })
        val r = NumpyRandom(0)
        for (p in rng["perms"]!!) {
            @Suppress("UNCHECKED_CAST")
            val expected = (p as List<Double>).map { it.toInt() }.toIntArray()
            assertContentEquals(expected, r.permutation(expected.size))
        }
    }

    @Test
    fun bilinearResizeMatchesOpenCV() {
        val img = f32("img_37x53x3.f32")
        @Suppress("UNCHECKED_CAST")
        for ((key, hw) in fx["linear"] as Map<String, List<Double>>) {
            val (h, w) = hw.map { it.toInt() }
            val out = ImageOps.resizeLinear(img, 53, 3, 0, 0, 53, 37, h, w)
            assertTrue(maxDiff(f32("lin_$key.f32"), out) < 1e-5f, "linear $key: ${maxDiff(f32("lin_$key.f32"), out)}")
            // Planar (NCHW) output of the same resize.
            val planar = ImageOps.resizeLinear(img, 53, 3, 0, 0, 53, 37, h, w, planar = true)
            for (c in 0 until 3) for (i in 0 until h * w) assertEquals(out[i * 3 + c], planar[c * h * w + i])
        }
    }

    @Test
    fun areaResizeMatchesOpenCV() {
        val big = f32("gray_250x422.f32")
        assertTrue(maxDiff(f32("area_36x64.f32"), ImageOps.resizeArea(big, 250, 422, 36, 64)) < 1e-5f)
        val crop = FloatArray(144 * 256) { big[(it / 256) * 422 + it % 256] }
        assertTrue(maxDiff(f32("area_crop_36x64.f32"), ImageOps.resizeArea(crop, 144, 256, 36, 64)) < 1e-5f)
    }

    @Test
    fun connectedComponentsMatchOpenCV() {
        val bin = res("binary_30x40.u8").map { it.toInt() != 0 }.toBooleanArray()
        val (n, labels) = ImageOps.connectedComponents(bin, 30, 40)
        @Suppress("UNCHECKED_CAST")
        val cc = fx["cc"] as Map<String, Any>
        assertEquals((cc["n"] as Double).toInt(), n)
        val sizes = (1 until n).map { k -> labels.count { it == k } }.sorted()
        assertEquals((cc["sizes"] as List<Double>).map { it.toInt() }, sizes)
    }

    @Test
    fun gridFractionMatchesPython() {
        val bits = res("mask_240x426.bits")
        val mask = Mask(240, 426, BooleanArray(240 * 426) { (bits[it shr 3].toInt() shr (7 - (it and 7))) and 1 == 1 })
        assertTrue(maxDiff(f32("frac_16x28.f32"), ImageOps.gridFraction(mask, 16, 28)) < 1e-6f)
    }

    @Test
    fun boxHelpers() {
        val m = ImageOps.boxMask(floatArrayOf(2f, 3f, 6f, 5f), 10, 10, pad = 0.15f)
        // p = 4 * 0.15 = 0.6 → rows int(2.4)=2 .. ceil(5.6)=6, cols int(1.4)=1 .. ceil(6.6)=7
        assertContentEquals(floatArrayOf(1f, 2f, 7f, 6f), ImageOps.bbox(m))
        assertEquals(1f / 7f, ImageOps.boxIou(floatArrayOf(0f, 0f, 2f, 2f), floatArrayOf(1f, 1f, 3f, 3f)), 1e-6f)
        assertEquals(listOf(0 to 2, 1 to 4, 3 to 5), ImageOps.poolBins(5, 3).map { it[0] to it[1] })
    }
}
