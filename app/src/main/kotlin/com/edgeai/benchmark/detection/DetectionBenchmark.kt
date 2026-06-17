package com.edgeai.benchmark.detection

import com.edgeai.benchmark.model.Backend
import com.edgeai.benchmark.model.Precision
import com.edgeai.benchmark.model.Runtime
import com.edgeai.benchmark.model.ThermalStatus

/** One detection run split into phases. All in milliseconds. */
data class PhaseTiming(
    val preprocessMs: Double,    // letterbox + RGB extract + normalize + (INT8) input quantize
    val inferenceMs: Double,     // interpreter.run only
    val postprocessMs: Double,   // output read + (INT8) dequant + decode + NMS
    val endToEndMs: Double,      // outer timer over all three (NOT a sum of the above)
    val detectionCount: Int
)

/**
 * Aggregated 3-phase detection benchmark for one (precision) config.
 * Written to a dedicated detection CSV (separate from the classification CSV,
 * whose schema is single-latency). Phases match REVIEW_PLAN_v0.5.3:
 *  - preprocess includes INT8 input quantization
 *  - inference is interpreter.run only
 *  - postprocess includes INT8 output dequant + decode + NMS
 *  - end-to-end is measured by its own outer timer (p50 of e2e ≠ sum of phase p50s)
 *  - rendering is NOT measured here (visualization only via Detect Sample)
 */
data class DetectionBenchmarkResult(
    val timestampUtcMs: Long,
    val modelName: String,
    val runtime: Runtime,
    val backend: Backend,
    val precision: Precision,
    val inputWidth: Int,
    val inputHeight: Int,
    val warmupRuns: Int,
    val measuredRuns: Int,

    val preprocessP50: Double, val preprocessP90: Double, val preprocessP99: Double,
    val inferenceP50: Double,  val inferenceP90: Double,  val inferenceP99: Double,
    val postprocessP50: Double, val postprocessP90: Double, val postprocessP99: Double,
    val endToEndP50: Double,   val endToEndP90: Double,   val endToEndP99: Double,

    val detectionCountP50: Int,
    val confThreshold: Float,
    val iouThreshold: Float,
    val maxDetections: Int,

    val modelSizeMb: Double,
    val memBeforeMb: Double,
    val memPeakMb: Double,
    val memAfterMb: Double,
    val thermalBefore: ThermalStatus,
    val thermalAfter: ThermalStatus,

    val deviceModel: String,
    val deviceChip: String,
    val androidVersion: Int,
    val abiName: String
) {
    fun toCsvRow(): String = listOf(
        timestampUtcMs, "detection", modelName, runtime.label, backend.label, precision.label,
        inputWidth, inputHeight, warmupRuns, measuredRuns,
        r(preprocessP50), r(preprocessP90), r(preprocessP99),
        r(inferenceP50), r(inferenceP90), r(inferenceP99),
        r(postprocessP50), r(postprocessP90), r(postprocessP99),
        r(endToEndP50), r(endToEndP90), r(endToEndP99),
        detectionCountP50, confThreshold, iouThreshold, maxDetections,
        r(modelSizeMb), r(memBeforeMb), r(memPeakMb), r(memAfterMb),
        thermalBefore.name, thermalAfter.name,
        deviceModel, deviceChip, androidVersion, abiName
    ).joinToString(",")

    private fun r(v: Double): Double = Math.round(v * 1000.0) / 1000.0
}

/** Floor-index nearest-rank percentile, matching BenchmarkEngine. */
object Percentiles {
    fun of(values: DoubleArray, pct: Double): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sortedArray()
        val idx = ((pct / 100.0) * (sorted.size - 1)).toInt().coerceIn(0, sorted.size - 1)
        return sorted[idx]
    }
}
