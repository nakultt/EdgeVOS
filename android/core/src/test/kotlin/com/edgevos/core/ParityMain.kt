package com.edgevos.core

import java.io.DataInputStream
import java.io.EOFException
import java.io.File

/**
 * JVM parity run: the Kotlin tracker on a video with the ONNX Runtime CPU EP, frames decoded by the
 * ffmpeg CLI exactly like android/tools/parity_reference.py. Writes masks.bin + events.tsv.
 *
 *   ./gradlew :core:parity --args="--video ../samples/b.mp4 --prompt cat --out build/parity/kt_cat"
 */
fun main(args: Array<String>) {
    val opts = args.toList().chunked(2).associate { (k, v) -> k.removePrefix("--") to v }
    val video = opts.getValue("video")
    val prompt = opts.getValue("prompt")
    val out = File(opts.getValue("out")).apply { mkdirs() }
    val modelDir = File(opts["models"] ?: "../mobile/models")
    val limit = opts["limit"]?.toInt() ?: Int.MAX_VALUE
    val fp32Text = opts["fp32-text"] == "1"

    val (h, w) = ProcessBuilder("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", video)
        .start().inputStream.bufferedReader().readText().trim().split(",").let { it[1].toInt() to it[0].toInt() }

    val manifest = Manifest.load(File(modelDir, "manifest.json"))
    val sessions = CpuSessions(modelDir, manifest, int8Models = if (fp32Text) emptySet() else setOf("clip_text"))
    val models = Models(manifest, sessions, Models.loadBackground(File(modelDir, "clip_background.npy")))
    val tracker = MobileTracker(models)

    val proc = ProcessBuilder("ffmpeg", "-v", "error", "-i", video, "-f", "rawvideo", "-pix_fmt", "rgb24", "-")
        .redirectError(ProcessBuilder.Redirect.INHERIT).start()
    val input = DataInputStream(proc.inputStream.buffered(1 shl 20))
    val buf = ByteArray(h * w * 3)
    val start = System.nanoTime()
    var i = 0
    MaskWriter(File(out, "masks.bin"), h, w).use { masks ->
        File(out, "events.tsv").bufferedWriter().use { ev ->
            while (i < limit) {
                try { input.readFully(buf) } catch (_: EOFException) { break }
                val frame = FloatImage.fromRgb(buf, h, w)
                val t0 = System.nanoTime()
                val mask = if (i == 0) tracker.init(frame, prompt) else tracker.step(frame)
                val ms = (System.nanoTime() - t0) / 1e6
                masks.write(mask)
                ev.write("$i\t${tracker.lastEvent ?: "-"}\t${"%.1f".format(ms)}\t${models.timings}\n")
                tracker.lastEvent?.let { println("$i $it") }
                i++
            }
        }
    }
    proc.destroy()
    println("$i frames in ${"%.0f".format((System.nanoTime() - start) / 1e9)} s")
    models.close()
}
