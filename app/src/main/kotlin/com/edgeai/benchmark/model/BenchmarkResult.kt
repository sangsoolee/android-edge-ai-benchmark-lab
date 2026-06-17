package com.edgeai.benchmark.model

/**
 * Immutable record of a single benchmark run.
 * One instance per (runtime × backend × model × precision) combination.
 */
data class BenchmarkResult(
    // --- Identity ---
    val runtime: Runtime,
    val backend: Backend,
    val modelName: String,
    val precision: Precision,

    // --- Model ---
    val modelSizeMb: Double,

    // --- Run config ---
    val warmupRuns: Int,
    val measuredRuns: Int,

    // --- Latency (ms) ---
    val p50LatencyMs: Double,
    val p95LatencyMs: Double,
    val p99LatencyMs: Double,
    val avgLatencyMs: Double,
    val minLatencyMs: Double,
    val maxLatencyMs: Double,
    val coldStartMs: Double,

    // --- Memory ---
    val peakMemoryMb: Double,

    // --- Thermal ---
    val thermalBefore: ThermalStatus,
    val thermalAfter: ThermalStatus,

    // --- Device context ---
    val deviceModel: String,
    val deviceChip: String,
    val androidVersion: Int,
    val androidBuildId: String,
    val abiName: String,

    // --- Timestamp ---
    val timestampUtcMs: Long,

    // --- v0.x additions (appended to preserve CSV column order) ---
    // What the per-inference timer covers. END_TO_END includes the runtime's
    // natural output handling; KERNEL is the bare compute call (see BenchmarkEngine).
    val latencyMode: LatencyMode = LatencyMode.END_TO_END,
    // PSS snapshots at phase boundaries (peakMemoryMb is the max during measured loop)
    val memAfterLoadMb: Double = 0.0,
    val memAfterWarmupMb: Double = 0.0,
    val memAfterMeasuredMb: Double = 0.0
) {
    /**
     * Serialize to CSV row. Column order matches [CsvExporter.HEADER].
     */
    fun toCsvRow(): String = listOf(
        timestampUtcMs,
        runtime.label,
        backend.label,
        modelName,
        precision.label,
        modelSizeMb,
        warmupRuns,
        measuredRuns,
        p50LatencyMs,
        p95LatencyMs,
        p99LatencyMs,
        avgLatencyMs,
        minLatencyMs,
        maxLatencyMs,
        coldStartMs,
        peakMemoryMb,
        thermalBefore.name,
        thermalAfter.name,
        deviceModel,
        deviceChip,
        androidVersion,
        androidBuildId,
        abiName,
        latencyMode.label,
        memAfterLoadMb,
        memAfterWarmupMb,
        memAfterMeasuredMb
    ).joinToString(",")
}

// ---------------------------------------------------------------------------

enum class Runtime(val label: String) {
    LITERT("LiteRT"),
    ONNX_RUNTIME("ONNXRuntime"),
    EXECUTORCH("ExecuTorch")
}

enum class Backend(val label: String) {
    CPU("CPU"),
    GPU_DELEGATE("GPU"),
    XNNPACK("XNNPACK"),
    NNAPI("NNAPI"),
    VULKAN("Vulkan")
}

enum class Precision(val label: String) {
    FP32("FP32"),
    FP16("FP16"),
    INT8("INT8"),
    INT4("INT4")
}

/**
 * What the per-inference latency timer covers.
 *
 *  - END_TO_END: the inference call as the runtime's app-level API naturally
 *    performs it, including output materialization. This is what an app actually
 *    pays per call, but it includes per-runtime wrapper overhead (e.g. ONNX/
 *    ExecuTorch allocate an output object per call; LiteRT writes into a reused
 *    buffer), so cross-runtime gaps partly reflect API overhead, not just kernels.
 *  - KERNEL: the bare compute call with I/O already bound, isolating raw kernel
 *    time for a fairer compute comparison.
 */
enum class LatencyMode(val label: String) {
    END_TO_END("end_to_end"),
    KERNEL("kernel")
}

/**
 * Maps to Android PowerManager thermal status constants.
 * @see android.os.PowerManager.THERMAL_STATUS_*
 */
enum class ThermalStatus(val androidCode: Int) {
    NONE(0),
    LIGHT(1),
    MODERATE(2),
    SEVERE(3),
    CRITICAL(4),
    EMERGENCY(5),
    SHUTDOWN(6),
    UNKNOWN(-1);

    companion object {
        fun fromAndroidCode(code: Int): ThermalStatus =
            entries.firstOrNull { it.androidCode == code } ?: UNKNOWN
    }
}
