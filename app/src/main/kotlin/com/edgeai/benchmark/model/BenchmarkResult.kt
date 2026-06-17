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
    val timestampUtcMs: Long
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
        abiName
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
