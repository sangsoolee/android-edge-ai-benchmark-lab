package com.edgeai.benchmark.util

import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import com.edgeai.benchmark.model.ThermalStatus
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

// ---------------------------------------------------------------------------
// ThermalMonitor
// ---------------------------------------------------------------------------

/**
 * Reads the current Android thermal status.
 * Requires API 29+ for THERMAL_STATUS_* constants.
 * Falls back to UNKNOWN on older devices.
 */
object ThermalMonitor {

    fun currentStatus(context: Context): ThermalStatus {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return ThermalStatus.UNKNOWN

        val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        return ThermalStatus.fromAndroidCode(pm.currentThermalStatus)
    }
}

// ---------------------------------------------------------------------------
// MemoryTracker
// ---------------------------------------------------------------------------

/**
 * Polls process PSS memory at [intervalMs] while inference runs.
 * Captures the peak value seen during the measurement window.
 *
 * Usage:
 *   tracker.start()
 *   ... run inference ...
 *   val peakMb = tracker.stopAndGetPeakMb()
 */
class MemoryTracker(
    private val context: Context,
    // Tightened from 50ms: short models (~1.5ms/inference, ~150ms/100 runs) only
    // gave a few samples at 50ms, so a transient peak could be missed.
    private val intervalMs: Long = 10L
) {
    private var executor: ScheduledExecutorService? = null
    private val peakBytes = AtomicReference(0L)

    fun start() {
        peakBytes.set(0L)
        executor = Executors.newSingleThreadScheduledExecutor()
        executor!!.scheduleAtFixedRate({
            val current = currentPssBytes()
            peakBytes.updateAndGet { prev -> maxOf(prev, current) }
        }, 0, intervalMs, TimeUnit.MILLISECONDS)
    }

    fun stopAndGetPeakMb(): Double {
        executor?.shutdown()
        executor?.awaitTermination(1, TimeUnit.SECONDS)
        executor = null
        return peakBytes.get() / (1024.0 * 1024.0)
    }

    /** Returns total PSS in bytes using Debug.MemoryInfo */
    private fun currentPssBytes(): Long {
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        // totalPss is in kB
        return info.totalPss * 1024L
    }

    companion object {
        /** One-shot PSS snapshot in MB, for recording at phase boundaries. */
        fun sampleNowMb(): Double {
            val info = Debug.MemoryInfo()
            Debug.getMemoryInfo(info)
            return info.totalPss * 1024L / (1024.0 * 1024.0)
        }
    }
}
