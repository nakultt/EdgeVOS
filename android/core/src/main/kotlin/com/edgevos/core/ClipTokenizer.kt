package com.edgevos.core

import java.io.InputStream
import java.util.Locale
import java.util.zip.GZIPInputStream

/**
 * CLIP BPE tokenizer (context 77), a port of mobile/tokenizer.py. The vocabulary is
 * mobile/assets/bpe_simple_vocab_16e6.txt.gz, packaged as a classpath resource.
 */
class ClipTokenizer(vocab: InputStream = defaultVocab(), private val contextLength: Int = 77) {
    private val byteOrder: List<Int> = byteOrder()
    private val byteEncoder: Array<String> = Array(256) { String(Character.toChars(unicodeFor(it))) }
    private val encoder = HashMap<String, Int>(1 shl 17)
    private val bpeRanks = HashMap<Pair<String, String>, Int>(1 shl 17)
    private val cache = HashMap<String, List<String>>()
    val sot: Int
    val eot: Int

    init {
        val lines = GZIPInputStream(vocab).bufferedReader(Charsets.UTF_8).use { it.readText() }.split("\n")
        val merges = lines.subList(1, 49152 - 256 - 2 + 1).map { val p = it.split(" "); p[0] to p[1] }
        val words = ArrayList<String>(49408)
        // Python builds the vocabulary from byte_encoder.values() in insertion (bs) order.
        byteOrder.mapTo(words) { byteEncoder[it] }
        byteOrder.mapTo(words) { byteEncoder[it] + "</w>" }
        merges.mapTo(words) { it.first + it.second }
        words += "<|startoftext|>"
        words += "<|endoftext|>"
        words.forEachIndexed { i, w -> encoder[w] = i }
        merges.forEachIndexed { i, m -> bpeRanks[m] = i }
        sot = encoder.getValue("<|startoftext|>")
        eot = encoder.getValue("<|endoftext|>")
        cache["<|startoftext|>"] = listOf("<|startoftext|>")
        cache["<|endoftext|>"] = listOf("<|endoftext|>")
    }

    private fun bpe(token: String): List<String> {
        cache[token]?.let { return it }
        // Split into code points (byte-encoded chars are all in the BMP, but stay safe).
        val chars = token.codePoints().toArray().map { String(Character.toChars(it)) }
        var word: List<String> = chars.dropLast(1) + (chars.last() + "</w>")
        if (word.size < 2) return listOf(token + "</w>")
        while (true) {
            var best: Pair<String, String>? = null
            var bestRank = Int.MAX_VALUE
            for (i in 0 until word.size - 1) {
                val r = bpeRanks[word[i] to word[i + 1]] ?: continue
                if (r < bestRank) { bestRank = r; best = word[i] to word[i + 1] }
            }
            if (best == null) break
            val (first, second) = best
            val merged = ArrayList<String>(word.size)
            var i = 0
            while (i < word.size) {
                val j = word.subList(i, word.size).indexOf(first).let { if (it < 0) -1 else it + i }
                if (j < 0) { merged.addAll(word.subList(i, word.size)); break }
                merged.addAll(word.subList(i, j))
                i = j
                if (word[i] == first && i < word.size - 1 && word[i + 1] == second) {
                    merged.add(first + second); i += 2
                } else {
                    merged.add(word[i]); i += 1
                }
            }
            word = merged
            if (word.size == 1) break
        }
        cache[token] = word
        return word
    }

    fun encode(text: String): List<Int> {
        val ids = ArrayList<Int>()
        val m = PATTERN.matcher(clean(text))
        while (m.find()) {
            val token = m.group().toByteArray(Charsets.UTF_8).joinToString("") { byteEncoder[it.toInt() and 0xff] }
            bpe(token).mapTo(ids) { encoder.getValue(it) }
        }
        return ids
    }

    /** Token ids [contextLength] (SOT ... EOT, zero padded), the model's int64 [1, 77] input. */
    operator fun invoke(text: String): LongArray {
        val ids = ArrayList<Int>().apply { add(sot); addAll(encode(text)); add(eot) }
        val out = LongArray(contextLength)
        val n = minOf(ids.size, contextLength)
        for (i in 0 until n) out[i] = ids[i].toLong()
        if (ids.size > contextLength) out[contextLength - 1] = eot.toLong()
        return out
    }

    companion object {
        // Python's \s / \S only meet single spaces here: clean() has already collapsed whitespace.
        private val PATTERN = java.util.regex.Pattern.compile(
            "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^ \\p{L}\\p{N}]+",
            java.util.regex.Pattern.CASE_INSENSITIVE or java.util.regex.Pattern.UNICODE_CASE,
        )

        fun defaultVocab(): InputStream =
            ClipTokenizer::class.java.classLoader!!.getResourceAsStream("bpe_simple_vocab_16e6.txt.gz")
                ?: error("bpe_simple_vocab_16e6.txt.gz not on the classpath")

        /** bytes_to_unicode(): the byte order of the dict (printable bytes first, then the rest). */
        private fun byteOrder(): List<Int> {
            val bs = ArrayList<Int>()
            bs.addAll('!'.code..'~'.code); bs.addAll('¡'.code..'¬'.code); bs.addAll('®'.code..'ÿ'.code)
            for (b in 0 until 256) if (b !in bs) bs += b
            return bs
        }

        /** bytes_to_unicode()[b]: printable bytes map to themselves, the others to 256 + n. */
        private fun unicodeFor(b: Int): Int {
            val order = byteOrder()
            val printable = 188  // 94 + 12 + 82
            val pos = order.indexOf(b)
            return if (pos < printable) b else 256 + (pos - printable)
        }

        /** str.isspace() */
        private fun isPySpace(c: Char) =
            c in "\t\n\u000b\u000c\r\u001c\u001d\u001e\u001f \u0085  " ||
                Character.getType(c) == Character.SPACE_SEPARATOR.toInt()

        private val NAMED = mapOf("amp" to "&", "lt" to "<", "gt" to ">", "quot" to "\"", "apos" to "'", "nbsp" to " ")

        /** html.unescape for numeric references and the common named entities. */
        private fun unescape(s: String): String {
            if ('&' !in s) return s
            return Regex("&(#[0-9]+|#[xX][0-9a-fA-F]+|[a-zA-Z]+);").replace(s) { m ->
                val e = m.groupValues[1]
                when {
                    e.startsWith("#x") || e.startsWith("#X") -> e.substring(2).toIntOrNull(16)?.let { String(Character.toChars(it)) }
                    e.startsWith("#") -> e.substring(1).toIntOrNull()?.let { String(Character.toChars(it)) }
                    else -> NAMED[e]
                } ?: m.value
            }
        }

        /** tokenizer._clean: unescape twice, collapse whitespace, strip, lower-case. */
        fun clean(text: String): String {
            val t = unescape(unescape(text))
            val b = StringBuilder()
            var space = false
            for (c in t) {
                if (isPySpace(c)) { space = true; continue }
                if (space && b.isNotEmpty()) b.append(' ')
                space = false
                b.append(c)
            }
            return b.toString().lowercase(Locale.ROOT)
        }
    }
}
