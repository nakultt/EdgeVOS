package com.edgevos.core

import java.io.BufferedOutputStream
import java.io.File
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Writes masks.bin (format in android/tools/masks.py): "EVM1", int32 LE H, W, then per frame a
 * presence byte followed by the mask packed MSB-first (numpy.packbits).
 */
class MaskWriter(file: File, private val h: Int, private val w: Int) : AutoCloseable {
    private val out: OutputStream = BufferedOutputStream(file.outputStream(), 1 shl 16)
    private val packed = ByteArray((h * w + 7) / 8)

    init {
        out.write("EVM1".toByteArray(Charsets.US_ASCII))
        out.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putInt(h).putInt(w).array())
    }

    fun write(mask: Mask?) {
        if (mask == null) { out.write(0); return }
        require(mask.h == h && mask.w == w)
        packed.fill(0)
        for (i in mask.data.indices) if (mask.data[i]) packed[i shr 3] = (packed[i shr 3].toInt() or (0x80 ushr (i and 7))).toByte()
        out.write(1)
        out.write(packed)
    }

    override fun close() = out.close()
}
