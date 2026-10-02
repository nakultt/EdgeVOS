package com.edgevos.core

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Minimal JSON reader for manifest.json → Map / List / Double / String / Boolean / null. */
object Json {
    fun parse(text: String): Any? = Parser(text).run { val v = value(); ws(); require(i == s.length) { "trailing data at $i" }; v }

    private class Parser(val s: String) {
        var i = 0

        fun ws() { while (i < s.length && s[i].isWhitespace()) i++ }

        fun value(): Any? {
            ws()
            return when (val c = s[i]) {
                '{' -> obj()
                '[' -> arr()
                '"' -> str()
                't' -> lit("true", true)
                'f' -> lit("false", false)
                'n' -> lit("null", null)
                else -> if (c == '-' || c.isDigit()) num() else error("unexpected '$c' at $i")
            }
        }

        fun lit(word: String, v: Any?): Any? {
            require(s.startsWith(word, i)) { "bad literal at $i" }
            i += word.length
            return v
        }

        fun obj(): Map<String, Any?> {
            val m = LinkedHashMap<String, Any?>()
            i++
            ws()
            if (s[i] == '}') { i++; return m }
            while (true) {
                ws()
                val k = str()
                ws(); require(s[i] == ':'); i++
                m[k] = value()
                ws()
                if (s[i] == ',') { i++; continue }
                require(s[i] == '}') { "expected } at $i" }; i++
                return m
            }
        }

        fun arr(): List<Any?> {
            val l = ArrayList<Any?>()
            i++
            ws()
            if (s[i] == ']') { i++; return l }
            while (true) {
                l.add(value())
                ws()
                if (s[i] == ',') { i++; continue }
                require(s[i] == ']') { "expected ] at $i" }; i++
                return l
            }
        }

        fun str(): String {
            require(s[i] == '"') { "expected string at $i" }
            i++
            val b = StringBuilder()
            while (true) {
                val c = s[i++]
                when (c) {
                    '"' -> return b.toString()
                    '\\' -> when (val e = s[i++]) {
                        'n' -> b.append('\n'); 't' -> b.append('\t'); 'r' -> b.append('\r')
                        'b' -> b.append('\b'); 'f' -> b.append('\u000c')
                        'u' -> { b.append(s.substring(i, i + 4).toInt(16).toChar()); i += 4 }
                        else -> b.append(e)
                    }
                    else -> b.append(c)
                }
            }
        }

        fun num(): Double {
            val start = i
            while (i < s.length && (s[i].isDigit() || s[i] in "+-.eE")) i++
            return s.substring(start, i).toDouble()
        }
    }
}

/** Reads a float32 matrix from a .npy file (little-endian '<f4', C order) or a raw float32 LE file. */
object FloatMatrix {
    fun read(input: InputStream, cols: Int): Array<FloatArray> {
        val bytes = input.readBytes()
        var offset = 0
        if (bytes.size >= 10 && bytes[0] == 0x93.toByte() && String(bytes, 1, 5, Charsets.US_ASCII) == "NUMPY") {
            val major = bytes[6].toInt()
            val hdrLen: Int
            if (major == 1) {
                hdrLen = (bytes[8].toInt() and 0xff) or ((bytes[9].toInt() and 0xff) shl 8); offset = 10 + hdrLen
            } else {
                hdrLen = ByteBuffer.wrap(bytes, 8, 4).order(ByteOrder.LITTLE_ENDIAN).int; offset = 12 + hdrLen
            }
            val header = String(bytes, offset - hdrLen, hdrLen, Charsets.US_ASCII)
            require("'<f4'" in header && "'fortran_order': False" in header) { "unsupported npy: $header" }
        }
        val buf = ByteBuffer.wrap(bytes, offset, bytes.size - offset).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val rows = buf.remaining() / cols
        return Array(rows) { FloatArray(cols).also { r -> buf.get(r) } }
    }
}
