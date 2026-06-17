package com.edgeai.benchmark.util

import android.content.Context
import com.edgeai.benchmark.model.BenchmarkResult
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Appends benchmark results to a dated CSV file in the app's external files directory.
 * Pull with: adb pull /sdcard/Android/data/com.edgeai.benchmark/files/results/
 */
object CsvExporter {

    val HEADER = listOf(
        "timestamp_utc_ms",
        "runtime", "backend", "model_name", "precision",
        "model_size_mb",
        "warmup_runs", "measured_runs",
        "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
        "avg_latency_ms", "min_latency_ms", "max_latency_ms",
        "cold_start_ms",
        "peak_memory_mb",
        "thermal_before", "thermal_after",
        "device_model", "device_chip", "android_version",
        "android_build_id", "abi",
        "latency_mode",
        "mem_after_load_mb", "mem_after_warmup_mb", "mem_after_measured_mb"
    ).joinToString(",")

    /**
     * Appends [result] to the current session CSV file.
     * Creates the file (with header) if it doesn't exist yet.
     *
     * @return the [File] written to
     */
    fun append(context: Context, result: BenchmarkResult): File {
        val dir = File(context.getExternalFilesDir(null), "results").apply { mkdirs() }
        val dateStr = SimpleDateFormat("yyyyMMdd", Locale.US).format(Date())
        val file = File(dir, "benchmark_$dateStr.csv")

        if (!file.exists()) {
            file.writeText(HEADER + "\n")
        }
        file.appendText(result.toCsvRow() + "\n")
        return file
    }

    /**
     * Writes all [results] to a single CSV file, overwriting if it exists.
     */
    fun writeAll(context: Context, results: List<BenchmarkResult>, filename: String): File {
        val dir = File(context.getExternalFilesDir(null), "results").apply { mkdirs() }
        val file = File(dir, filename)
        val sb = StringBuilder(HEADER).append("\n")
        results.forEach { sb.append(it.toCsvRow()).append("\n") }
        file.writeText(sb.toString())
        return file
    }
}
