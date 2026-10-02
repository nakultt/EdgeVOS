import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// The tracker port: plain Kotlin/JVM so it runs (and is parity-checked) on the desktop JVM with
// ONNX Runtime's CPU EP and is used unchanged by the Android app, whose onnxruntime-android-qnn
// AAR provides the same ai.onnxruntime API.
plugins {
    kotlin("jvm")
}

val ortVersion = "1.29.0"

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

sourceSets {
    // The CLIP BPE vocabulary ships with the Python tokenizer; package it as a resource.
    main { resources.srcDir("../../mobile/assets") }
}

dependencies {
    compileOnly("com.microsoft.onnxruntime:onnxruntime:$ortVersion")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:$ortVersion")
    testImplementation(kotlin("test"))
    testImplementation("junit:junit:4.13.2")
}

tasks.test {
    maxHeapSize = "2g"
    testLogging { events("passed", "failed"); showStandardStreams = true }
}

// JVM parity run against mobile/runtime.py:
//   ./gradlew :core:parity --args="--video ../samples/b.mp4 --prompt cat --out build/parity/kt_cat"
tasks.register<JavaExec>("parity") {
    group = "verification"
    description = "Run the Kotlin tracker on a video with the ORT CPU EP and dump masks/events."
    classpath = sourceSets["test"].runtimeClasspath
    mainClass.set("com.edgevos.core.ParityMainKt")
    workingDir = rootProject.projectDir
    maxHeapSize = "4g"
}
