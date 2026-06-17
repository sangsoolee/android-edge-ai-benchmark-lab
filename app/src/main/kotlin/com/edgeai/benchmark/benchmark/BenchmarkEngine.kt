package com.edgeai.benchmark.benchmark

import android.content.Context
import android.os.SystemClock
import com.edgeai.benchmark.model.BenchmarkResult
import com.edgeai.benchmark.model.Backend
import com.edgeai.benchmark.model.LatencyMode
import com.edgeai.benchmark.model.Precision
import com.edgeai.benchmark.model.Runtime
import com.edgeai.benchmark.util.MemoryTracker
import com.edgeai.benchmark.util.ThermalMonitor

/**
 * Orchestrates a full benchmark run:
 *   1. warm-up phase (discarded)
 *   2. measurement phase (recorded)
 *   3. stats aggregation → BenchmarkResult
 *
 * Implementations must provide [loadModel], [runInference], [unloadModel].
 */
abstract class BenchmarkEngine(protected val context: Context) {

    abstract val runtime: Runtime
    abstract val backend: Backend

    /**
     * Load (or reload) the model from [modelPath].
     * Must be called before [benchmark].
     * @return cold-start time in milliseconds
     */
    abstract fun loadModel(modelPath: String, precision: Precision): Double

    /**
     * Execute a single forward pass on a fixed synthetic input, timed the way the
     * runtime's app-level API naturally works (END_TO_END — includes output handling).
     * @return inference time in milliseconds
     */
    abstract fun runInference(): Double

    /**
     * Bare compute call with I/O already bound (KERNEL mode). Override per engine
     * to isolate raw kernel time; the default falls back to [runInference] so a
     * KERNEL run is never *slower* than END_TO_END and engines opt in incrementally.
     */
    open fun runInferenceKernel(): Double = runInference()

    /** Release native resources. */
    abstract fun unloadModel()

    // ------------------------------------------------------------------

    /**
     * Run a full benchmark session and return an aggregated [BenchmarkResult].
     *
     * @param modelPath   absolute path to the model file on device
     * @param modelName   human-readable model name (e.g. "MobileNetV3-Small")
     * @param precision   weight precision of the model file
     * @param warmupRuns  number of discarded warm-up iterations
     * @param measuredRuns  number of recorded iterations
     */
    fun benchmark(
        modelPath: String,
        modelName: String,
        precision: Precision,
        warmupRuns: Int = 20,
        measuredRuns: Int = 100,
        latencyMode: LatencyMode = LatencyMode.END_TO_END
    ): BenchmarkResult {
        val thermalBefore = ThermalMonitor.currentStatus(context)

        // 1. Cold start (load + first inference)
        val coldStartMs = loadModel(modelPath, precision)
        val memAfterLoadMb = MemoryTracker.sampleNowMb()

        // 2. Warm-up (discard results)
        val inferenceFn: () -> Double =
            if (latencyMode == LatencyMode.KERNEL) ::runInferenceKernel else ::runInference
        repeat(warmupRuns) { inferenceFn() }
        val memAfterWarmupMb = MemoryTracker.sampleNowMb()

        // 3. Measure
        val memoryTracker = MemoryTracker(context)
        memoryTracker.start()

        val latencies = DoubleArray(measuredRuns)
        repeat(measuredRuns) { i ->
            latencies[i] = inferenceFn()
        }

        val peakMemoryMb = memoryTracker.stopAndGetPeakMb()
        val memAfterMeasuredMb = MemoryTracker.sampleNowMb()
        val thermalAfter = ThermalMonitor.currentStatus(context)

        unloadModel()

        // 4. Compute stats
        latencies.sort()

        return BenchmarkResult(
            runtime         = runtime,
            backend         = backend,
            modelName       = modelName,
            precision       = precision,
            modelSizeMb     = modelSizeOnDisk(modelPath),
            warmupRuns      = warmupRuns,
            measuredRuns    = measuredRuns,
            p50LatencyMs    = percentile(latencies, 50.0),
            p95LatencyMs    = percentile(latencies, 95.0),
            p99LatencyMs    = percentile(latencies, 99.0),
            avgLatencyMs    = latencies.average(),
            minLatencyMs    = latencies.first(),
            maxLatencyMs    = latencies.last(),
            coldStartMs     = coldStartMs,
            peakMemoryMb    = peakMemoryMb,
            thermalBefore   = thermalBefore,
            thermalAfter    = thermalAfter,
            deviceModel     = android.os.Build.MODEL,
            deviceChip      = android.os.Build.HARDWARE,
            androidVersion  = android.os.Build.VERSION.SDK_INT,
            androidBuildId  = android.os.Build.ID,
            abiName         = android.os.Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown",
            timestampUtcMs  = System.currentTimeMillis(),
            latencyMode       = latencyMode,
            memAfterLoadMb    = memAfterLoadMb,
            memAfterWarmupMb  = memAfterWarmupMb,
            memAfterMeasuredMb = memAfterMeasuredMb
        )
    }

    // ------------------------------------------------------------------
    // Helpers

    private fun percentile(sorted: DoubleArray, pct: Double): Double {
        if (sorted.isEmpty()) return 0.0
        val idx = ((pct / 100.0) * (sorted.size - 1)).toInt().coerceIn(0, sorted.size - 1)
        return sorted[idx]
    }

    private fun modelSizeOnDisk(path: String): Double {
        val bytes = java.io.File(path).length()
        return bytes / (1024.0 * 1024.0)
    }
}
