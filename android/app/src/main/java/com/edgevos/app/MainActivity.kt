package com.edgevos.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.View
import android.view.WindowManager
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

/**
 * Live camera → EdgeVOS tracker. Type an object name and press Start: the prompt is grounded on the
 * current frame, then tracked every frame. The mask is drawn green on (re-)grounded frames and red on
 * tracked frames; the HUD shows fps, per-frame ms, the tracker event and the backend.
 */
class MainActivity : ComponentActivity() {
    private lateinit var preview: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var hud: TextView
    private lateinit var prompt: EditText
    private lateinit var startStop: Button
    private lateinit var loading: View

    private val loader = Executors.newSingleThreadExecutor()
    private val analysis = Executors.newSingleThreadExecutor()
    private val converter = YuvConverter(Engine.FRAME_LONG_SIDE)

    // Shared with the analysis thread.
    @Volatile private var running = false
    @Volatile private var activePrompt = ""
    @Volatile private var pendingInit = false
    @Volatile private var ready = false

    private var lastFrameNs = 0L
    private var fps = 0.0
    private var frames = 0

    private val permission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else Toast.makeText(this, R.string.camera_denied, Toast.LENGTH_LONG).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        preview = findViewById(R.id.preview)
        overlay = findViewById(R.id.overlay)
        hud = findViewById(R.id.hud)
        prompt = findViewById(R.id.prompt)
        startStop = findViewById(R.id.startStop)
        loading = findViewById(R.id.loading)
        preview.scaleType = PreviewView.ScaleType.FILL_CENTER

        startStop.setOnClickListener { if (running) stop() else start() }
        prompt.setOnEditorActionListener { _, action, _ ->
            if (action == EditorInfo.IME_ACTION_DONE) {
                hideKeyboard()
                // A new prompt restarts tracking; on a stopped tracker Done starts it.
                if (!running || prompt.text.toString().trim() != activePrompt) start()
                true
            } else false
        }

        loadModels()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else permission.launch(Manifest.permission.CAMERA)
    }

    private fun loadModels() {
        val first = Engine.models == null && Engine.needsOptimizing(this)
        findViewById<TextView>(R.id.loadingText).setText(if (first) R.string.optimizing else R.string.loading)
        val detail = findViewById<TextView>(R.id.loadingDetail)
        loader.execute {
            try {
                Engine.load(this) { msg -> runOnUiThread { detail.text = msg } }
                runOnUiThread {
                    loading.visibility = View.GONE
                    startStop.isEnabled = true
                    hud.text = "ready · ${Engine.backendLabel} · startup ${Engine.startupMs} ms"
                    ready = true
                }
            } catch (e: Throwable) {
                Log.e(TAG, "model loading failed", e)
                runOnUiThread { detail.text = "Failed to load models:\n${e.message}" }
            }
        }
    }

    private fun start() {
        val text = prompt.text.toString().trim()
        if (text.isEmpty() || !ready) return
        activePrompt = text
        pendingInit = true
        running = true
        startStop.setText(R.string.stop)
        hideKeyboard()
    }

    private fun stop() {
        running = false
        startStop.setText(R.string.start)
        overlay.clear()
        hud.text = "stopped · ${Engine.backendLabel}"
    }

    private fun hideKeyboard() {
        getSystemService(InputMethodManager::class.java).hideSoftInputFromWindow(prompt.windowToken, 0)
        prompt.clearFocus()
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            // Same 16:9 aspect for preview and analysis so the overlay lines up with the preview.
            val selector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(ResolutionStrategy(Size(1280, 720), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .build()
            val previewUse = Preview.Builder().setResolutionSelector(selector).build()
                .also { it.surfaceProvider = preview.surfaceProvider }
            val analysisUse = ImageAnalysis.Builder()
                .setResolutionSelector(selector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
            analysisUse.setAnalyzer(analysis, ::analyze)
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, previewUse, analysisUse)
        }, ContextCompat.getMainExecutor(this))
    }

    /** Analysis thread: convert, track, post the result. Frames arriving meanwhile are dropped. */
    private fun analyze(image: ImageProxy) {
        if (!ready || !running) { image.close(); return }
        val t0 = System.nanoTime()
        val frame = try { converter.convert(image) } finally { image.close() }
        val convMs = (System.nanoTime() - t0) / 1e6
        val initPrompt = if (pendingInit) { pendingInit = false; activePrompt } else null
        val r = try {
            Engine.process(frame, initPrompt)
        } catch (e: Throwable) {
            Log.e(TAG, "tracker failed", e)
            return
        }
        val now = System.nanoTime()
        val totalMs = (now - t0) / 1e6
        if (lastFrameNs != 0L && initPrompt == null) {
            val inst = 1e9 / (now - lastFrameNs)
            fps = if (fps == 0.0) inst else 0.9 * fps + 0.1 * inst
        }
        lastFrameNs = now
        frames++
        if (frames % 30 == 0 || r.event != "track") {
            Log.i(TAG, "frame ${frames} ${frame.w}x${frame.h} fps=%.1f total=%.1fms conv=%.1fms track=%.1fms event=%s %s"
                .format(fps, totalMs, convMs, r.trackMs, r.event, r.stages))
        }
        val text = "fps %.1f · %.1f ms\n%s · %s\n%s conv=%.1f".format(fps, totalMs, r.event, Engine.backendLabel, r.stages, convMs)
        runOnUiThread {
            if (!running) return@runOnUiThread
            overlay.show(r.mask, r.grounded)
            hud.text = text
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        running = false
        analysis.shutdown()
        loader.shutdown()
    }

    companion object {
        private const val TAG = "EdgeVOS"
    }
}
