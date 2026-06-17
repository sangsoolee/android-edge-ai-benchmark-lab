package com.edgeai.benchmark.detection

import android.content.Context
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Appends 3-phase detection benchmark rows to a dedicated CSV
 * (separate from the single-latency classification CSV).
 * Pull with: adb pull .../files/results/
 */
object DetectionCsvExporter {

    val HEADER = listOf(
        "timestamp_utc_ms", "benchmark_type", "model_name", "runtime", "backend", "precision",
        "input_width", "input_height", "warmup_runs", "measured_runs",
        "preprocess_p50_ms", "preprocess_p90_ms", "preprocess_p99_ms",
        "inference_p50_ms", "inference_p90_ms", "inference_p99_ms",
        "postprocess_p50_ms", "postprocess_p90_ms", "postprocess_p99_ms",
        "end_to_end_p50_ms", "end_to_end_p90_ms", "end_to_end_p99_ms",
        "detection_count_p50", "conf_threshold", "iou_threshold", "max_detections",
        "model_size_mb", "mem_before_mb", "mem_peak_mb", "mem_after_mb",
        "thermal_before", "thermal_after",
        "device_model", "device_chip", "android_version", "abi"
    ).joinToString(",")

    fun append(context: Context, result: DetectionBenchmarkResult): File {
        val dir = File(context.getExternalFilesDir(null), "results").apply { mkdirs() }
        val dateStr = SimpleDateFormat("yyyyMMdd", Locale.US).format(Date())
        val file = File(dir, "benchmark_detection_$dateStr.csv")
        if (!file.exists()) file.writeText(HEADER + "\n")
        file.appendText(result.toCsvRow() + "\n")
        return file
    }
}
