import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import java.nio.ByteBuffer
import java.nio.ByteOrder

plugins {
    id("com.android.application")
    kotlin("android")
}

android {
    namespace = "com.edgevos.app"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.edgevos.app"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        // Hexagon NPU + the measured ORT build are arm64 only.
        ndk { abiFilters += "arm64-v8a" }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    packaging {
        jniLibs {
            // Extract the QNN libraries to nativeLibraryDir so the Hexagon DSP loader can find the skel.
            useLegacyPackaging = true
            // qnn-runtime ships every Hexagon version plus the DSP/GPU backends; the S24 (SM8650) is HTP V75.
            excludes += listOf(
                "**/libQnnDsp.so", "**/libQnnDspV66Skel.so", "**/libQnnDspV66Stub.so", "**/libQnnGpu.so",
                "**/libQnnHtpV68Skel.so", "**/libQnnHtpV68Stub.so", "**/libQnnHtpV69Skel.so", "**/libQnnHtpV69Stub.so",
                "**/libQnnHtpV73Skel.so", "**/libQnnHtpV73Stub.so", "**/libQnnHtpV79Skel.so", "**/libQnnHtpV79Stub.so",
                "**/libQnnHtpV81Skel.so", "**/libQnnHtpV81Stub.so",
            )
        }
    }

    androidResources {
        // Models are memory-mapped/copied from the APK as-is; compressing 300 MB of weights gains nothing.
        noCompress += listOf("onnx", "f32", "json", "txt")
    }

    sourceSets["main"].assets.srcDir(layout.buildDirectory.dir("generated/modelAssets"))
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

dependencies {
    implementation(project(":core"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.29.0")
    implementation("com.qualcomm.qti:qnn-runtime:2.42.0")

    val camerax = "1.5.3"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")
    implementation("androidx.activity:activity-ktx:1.10.1")
    implementation("androidx.core:core-ktx:1.16.0")

    testImplementation("junit:junit:4.13.2")
}

// ── Model assets ────────────────────────────────────────────────────────────────────────────
// Copies the models the app uses from mobile/models (or -PedgevosModels=<dir>) into the APK's
// assets/models/, converting clip_background.npy to raw float32. The app copies them to filesDir on
// first launch. Build with -PbundleModels=false for a small APK and push the models with adb instead
// (see android/README.md).
val modelsDir = file(providers.gradleProperty("edgevosModels").getOrElse("../../mobile/models"))
val bundleModels = providers.gradleProperty("bundleModels").getOrElse("true").toBoolean()
val modelFiles = listOf(
    "manifest.json", "dinov3_patches.onnx", "dinov3_patches.int8.onnx", "segmenter.onnx",
    "clip_dense.onnx", "clip_text.int8.onnx",
)

val prepareModelAssets by tasks.registering {
    val out = layout.buildDirectory.dir("generated/modelAssets/models")
    if (bundleModels) {
        inputs.files(modelFiles.map { File(modelsDir, it) } + File(modelsDir, "clip_background.npy"))
    }
    inputs.property("bundleModels", bundleModels)
    outputs.dir(out)
    doLast {
        val dir = out.get().asFile
        dir.deleteRecursively()
        dir.mkdirs()
        if (!bundleModels) return@doLast
        val index = StringBuilder()
        for (name in modelFiles) {
            val src = File(modelsDir, name)
            require(src.exists()) { "missing $src: download the mobile-models-v1 release into mobile/models (android/README.md)" }
            src.copyTo(File(dir, name))
            index.append(name).append('\t').append(src.length()).append('\n')
        }
        // clip_background.npy → raw little-endian float32 [10, 512].
        val npy = File(modelsDir, "clip_background.npy").readBytes()
        val major = npy[6].toInt()
        val offset = if (major == 1) {
            10 + ((npy[8].toInt() and 0xff) or ((npy[9].toInt() and 0xff) shl 8))
        } else {
            12 + ByteBuffer.wrap(npy, 8, 4).order(ByteOrder.LITTLE_ENDIAN).int
        }
        val raw = npy.copyOfRange(offset, npy.size)
        File(dir, "clip_background.f32").writeBytes(raw)
        index.append("clip_background.f32\t").append(raw.size).append('\n')
        File(dir, "index.txt").writeText(index.toString())
    }
}

tasks.named("preBuild") { dependsOn(prepareModelAssets) }
