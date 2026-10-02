package com.edgevos.core

/**
 * Bit-exact port of `numpy.random.default_rng(seed).permutation(n)`: SeedSequence → PCG64
 * (XSL-RR 128/64) → Generator.shuffle (random_interval). PatchMemory subsamples its banks with it,
 * so the Kotlin tracker keeps exactly the patches the Python reference keeps.
 */
class NumpyRandom(seed: Long) {
    // 128-bit state and increment as (hi, lo) unsigned 64-bit halves.
    private var stateHi = 0L
    private var stateLo = 0L
    private var incHi = 0L
    private var incLo = 0L
    private var hasUint32 = false
    private var uinteger = 0

    init {
        val words = seedSequenceState(seed, 8)        // generate_state(4, uint64) as 8 uint32
        val v = LongArray(4) { (words[2 * it].toLong() and MASK32) or (words[2 * it + 1].toLong() shl 32) }
        // pcg64_set_seed: PCG_128BIT_CONSTANT(seed[0], seed[1]) is (hi, lo); same for inc.
        val initHi = v[0]; val initLo = v[1]
        val seqHi = v[2]; val seqLo = v[3]
        // pcg_setseq_128_srandom_r
        stateHi = 0; stateLo = 0
        incHi = (seqHi shl 1) or (seqLo ushr 63)
        incLo = (seqLo shl 1) or 1L
        step()
        val lo = stateLo + initLo
        stateHi = stateHi + initHi + if (java.lang.Long.compareUnsigned(lo, stateLo) < 0) 1 else 0
        stateLo = lo
        step()
    }

    /** state = state * MULT + inc (mod 2^128). */
    private fun step() {
        val lo = stateLo * MULT_LO
        val hi = umulh(stateLo, MULT_LO) + stateLo * MULT_HI + stateHi * MULT_LO
        val newLo = lo + incLo
        val carry = if (java.lang.Long.compareUnsigned(newLo, lo) < 0) 1L else 0L
        stateHi = hi + incHi + carry
        stateLo = newLo
    }

    fun nextLong(): Long {
        step()
        val x = stateHi xor stateLo
        val rot = (stateHi ushr 58).toInt()
        return java.lang.Long.rotateRight(x, rot)
    }

    fun nextInt(): Int {
        if (hasUint32) { hasUint32 = false; return uinteger }
        val next = nextLong()
        hasUint32 = true
        uinteger = (next ushr 32).toInt()
        return next.toInt()
    }

    /** numpy random_interval: uniform in [0, max] by masked rejection. */
    fun interval(max: Long): Long {
        if (max == 0L) return 0
        var mask = max
        mask = mask or (mask ushr 1); mask = mask or (mask ushr 2); mask = mask or (mask ushr 4)
        mask = mask or (mask ushr 8); mask = mask or (mask ushr 16); mask = mask or (mask ushr 32)
        if (max <= 0xffffffffL) {
            while (true) { val v = nextInt().toLong() and MASK32 and mask; if (v <= max) return v }
        } else {
            while (true) { val v = nextLong() and mask; if (v <= max) return v }
        }
    }

    /** Generator.permutation(n). */
    fun permutation(n: Int): IntArray {
        val a = IntArray(n) { it }
        for (i in n - 1 downTo 1) {
            val j = interval(i.toLong()).toInt()
            val t = a[i]; a[i] = a[j]; a[j] = t
        }
        return a
    }

    companion object {
        private const val MASK32 = 0xffffffffL
        private const val MULT_HI = 2549297995355413924L        // 0x2360ED051FC65DA4
        private const val MULT_LO = 4865540595714422341L        // 0x4385DF649FCCF645

        private const val INIT_A = 0x43b0d7e5
        private const val MULT_A = 0x931e8875.toInt()
        private const val INIT_B = 0x8b51f9dd.toInt()
        private const val MULT_B = 0x58f38ded
        private const val MIX_MULT_L = 0xca01f9dd.toInt()
        private const val MIX_MULT_R = 0x4973f715
        private const val POOL_SIZE = 4

        /** High 64 bits of the unsigned 128-bit product (Math.unsignedMultiplyHigh is not on Android). */
        private fun umulh(a: Long, b: Long): Long {
            val a0 = a and MASK32; val a1 = a ushr 32
            val b0 = b and MASK32; val b1 = b ushr 32
            val p00 = a0 * b0
            val p01 = a0 * b1
            val p10 = a1 * b0
            val mid = (p00 ushr 32) + (p01 and MASK32) + (p10 and MASK32)
            return a1 * b1 + (p01 ushr 32) + (p10 ushr 32) + (mid ushr 32)
        }

        /** numpy SeedSequence(seed).generate_state(nWords, uint32) for a non-negative integer seed. */
        fun seedSequenceState(seed: Long, nWords: Int): IntArray {
            require(seed >= 0)
            val entropy = ArrayList<Int>()
            var s = seed
            if (s == 0L) entropy += 0
            while (s > 0) { entropy += (s and MASK32).toInt(); s = s ushr 32 }

            var hashConst = INIT_A
            fun hashmix(value: Int): Int {
                var v = value xor hashConst
                hashConst *= MULT_A
                v *= hashConst
                return v xor (v ushr 16)
            }
            fun mix(x: Int, y: Int): Int {
                val r = MIX_MULT_L * x - MIX_MULT_R * y
                return r xor (r ushr 16)
            }
            val pool = IntArray(POOL_SIZE) { hashmix(if (it < entropy.size) entropy[it] else 0) }
            for (src in 0 until POOL_SIZE) for (dst in 0 until POOL_SIZE)
                if (src != dst) pool[dst] = mix(pool[dst], hashmix(pool[src]))
            for (src in POOL_SIZE until entropy.size) for (dst in 0 until POOL_SIZE)
                pool[dst] = mix(pool[dst], hashmix(entropy[src]))

            var hb = INIT_B
            return IntArray(nWords) { i ->
                var v = pool[i % POOL_SIZE] xor hb
                hb *= MULT_B
                v *= hb
                v xor (v ushr 16)
            }
        }
    }
}
