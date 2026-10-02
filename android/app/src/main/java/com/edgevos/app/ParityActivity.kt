package com.edgevos.app

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import com.edgevos.core.FloatImage
import com.edgevos.core.MaskWriter
import com.edgevos.core.fromRgb
import java.io.DataInputStream
import java.io.EOFException
import java.io.File
import kotlin.concurrent.thread

/**
 * Debug parity mode: runs the tracker on a video's frames and writes masks.bin + events.tsv (same
 * format as android/tools/parity_reference.py) to <external files>/parity/<name>/.
 *
 *   adb shell am start -n com.edgevos.app/.ParityActivity --es prompt cat \
 *       --es raw /sdcard/Android/data/com.edgevos.app/files/b.rgb --ei width 852 --ei height 480
 *   adb shell am start -n com.edgevos.app/.ParityActivity --es prompt cat \
 *       --es video /sdcard/Android/data/com.edgevos.app/files/b.mp4
 *
 * `raw` is rgb24 from `ffmpeg -i b.mp4 -f rawvideo -pix_fmt rgb24 b.rgb`: byte-identical to the frames
 * the Python reference sees. `video` decodes on the phone (MediaMetadataRetriever), whose YUV→RGB
 * differs slightly from ffmpeg. Optional: --ei frames N, --es name <output dir name>.
 */
class ParityActivity : ComponentActivity() {
    private lateinit var log: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        log = TextView(this).apply { setPadding(32, 96, 32, 32); typeface = android.graphics.Typeface.MONOSPACE; textSize = 11f }
        setContentView(ScrollView(this).apply { addView(log) })

        val prompt = intent.getStringExtra("prompt") ?: "cat"
        val raw = intent.getStringExtra("raw")
        val video = intent.getStringExtra("video")
        val limit = intent.getIntExtra("frames", Int.MAX_VALUE)
        val name = intent.getStringExtra("name") ?: prompt
        val out = File(getExternalFilesDir(null), "parity/$name").apply { mkdirs() }

        thread(name = "parity") {
            try {
                Engine.load(this) { say(it) }
                say("models ready in ${Engine.startupMs} ms on ${Engine.backendLabel}")
                val source = when {
                    raw != null -> rawFrames(File(raw), intent.getIntExtra("width", 852), intent.getIntExtra("height", 480))
                    video != null -> videoFrames(File(video))
                    else -> error("pass --es raw <rgb24 file> (with --ei width/height) or --es video <mp4>")
                }
                run(source, prompt, out, limit)
            } catch (e: Throwable) {
                Log.e(TAG, "parity failed", e)
                say("FAILED: $e")
            }
        }
    }

    private fun say(msg: String) {
        Log.i(TAG, msg)
        runOnUiThread { log.append(msg + "\n") }
    }

    private class FrameSource(val h: Int, val w: Int, val next: () -> FloatImage?)

    private fun rawFrames(file: File, w: Int, h: Int): FrameSource {
        val input = DataInputStream(file.inputStream().buffered(1 shl 20))
        val buf = ByteArray(w * h * 3)
        return FrameSource(h, w) {
            try { input.readFully(buf); FloatImage.fromRgb(buf, h, w) } catch (_: EOFException) { input.close(); null }
        }
    }

    private fun videoFrames(file: File): FrameSource {
        val r = MediaMetadataRetriever().apply { setDataSource(file.path) }
        val n = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)!!.toInt()
        val w = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)!!.toInt()
        val h = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)!!.toInt()
        var i = 0
        var batch: List<Bitmap> = emptyList()
        var bi = 0
        val px = IntArray(w * h)
        return FrameSource(h, w) {
            if (bi == batch.size) {
                batch = if (i < n) r.getFramesAtIndex(i, minOf(32, n - i)) else emptyList()
                bi = 0
            }
            if (bi == batch.size) { r.release(); null } else {
                val bmp = batch[bi++]
                i++
                bmp.getPixels(px, 0, w, 0, 0, w, h)
                bmp.recycle()
                val f = FloatArray(w * h * 3)
                for (p in px.indices) {
                    val c = px[p]
                    f[3 * p] = ((c shr 16) and 0xff) / 255f
                    f[3 * p + 1] = ((c shr 8) and 0xff) / 255f
                    f[3 * p + 2] = (c and 0xff) / 255f
                }
                FloatImage(h, w, 3, f)
            }
        }
    }

    private fun run(src: FrameSource, prompt: String, out: File, limit: Int) {
        say("parity: prompt '$prompt', ${src.w}x${src.h} → $out")
        var i = 0
        var trackMs = 0.0
        val t0 = System.nanoTime()
        MaskWriter(File(out, "masks.bin"), src.h, src.w).use { masks ->
            File(out, "events.tsv").bufferedWriter().use { ev ->
                while (i < limit) {
                    val frame = src.next() ?: break
                    val r = Engine.process(frame, if (i == 0) prompt else null)
                    masks.write(r.mask)
                    val event = Engine.tracker!!.lastEvent ?: "-"
                    ev.write("$i\t$event\t${"%.1f".format(r.trackMs)}\t${r.stages}\n")
                    trackMs += r.trackMs
                    if (event != "-") say("$i $event")
                    i++
                }
            }
        }
        val wall = (System.nanoTime() - t0) / 1e9
        say("DONE $i frames, tracker %.1f ms/frame mean (%.1f s wall incl. decoding). Pull with:\n  adb pull $out"
            .format(trackMs / maxOf(i, 1), wall))
    }

    companion object {
        private const val TAG = "EdgeVOS-Parity"
    }
}
