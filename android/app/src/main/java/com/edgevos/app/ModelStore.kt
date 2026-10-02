package com.edgevos.app

import android.content.Context
import android.util.Log
import java.io.File

/**
 * Where the models live on the phone.
 *
 * Default: the APK bundles them as uncompressed assets (assets/models/, see app/build.gradle.kts),
 * copied to filesDir/models on first launch because ONNX Runtime and QNN need real files.
 * Developer override: if <external files dir>/models/manifest.json exists (pushed with adb, for an
 * APK built with -PbundleModels=false), that directory is used as is.
 */
object ModelStore {
    private const val TAG = "EdgeVOS"

    fun externalDir(context: Context) = File(context.getExternalFilesDir(null), "models")

    /** Returns the model directory, copying bundled assets on first use. */
    fun resolve(context: Context, progress: (String) -> Unit): File {
        val external = externalDir(context)
        if (File(external, "manifest.json").exists()) {
            Log.i(TAG, "using pushed models in $external")
            return external
        }
        val index = try {
            context.assets.open("models/index.txt").bufferedReader().readLines().filter { it.isNotBlank() }
                .map { it.split('\t').let { p -> p[0] to p[1].toLong() } }
        } catch (e: java.io.IOException) {
            throw IllegalStateException(
                "No models: this APK was built without them. Push mobile/models to $external (android/README.md).", e,
            )
        }
        val dir = File(context.filesDir, "models").apply { mkdirs() }
        for ((name, size) in index) {
            val dst = File(dir, name)
            if (dst.length() == size) continue
            progress("Unpacking $name (${size / 1_000_000} MB)")
            val tmp = File(dir, "$name.part")
            context.assets.open("models/$name").use { input -> tmp.outputStream().use { input.copyTo(it, 1 shl 20) } }
            check(tmp.length() == size) { "short copy of $name" }
            tmp.renameTo(dst)
        }
        return dir
    }
}
