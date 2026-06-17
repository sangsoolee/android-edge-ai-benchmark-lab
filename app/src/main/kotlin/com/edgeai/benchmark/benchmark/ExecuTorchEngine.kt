package com.edgeai.benchmark.benchmark

import android.content.Context
import android.os.SystemClock
import com.edgeai.benchmark.model.Backend
import com.edgeai.benchmark.model.Precision
import com.edgeai.benchmark.model.Runtime
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

/**
 * Wraps ExecuTorch Android runtime for benchmarking.
 *
 * Model format: .pte (produced by export_executorch.py)
 * Input layout: NCHW float32 (1×3×224×224), fixed synthetic tensor allocated once.
 */
class ExecuTorchEngine(
    context: Context,
    requestedBackend: Backend = Backend.CPU
) : BenchmarkEngine(context) {

    init {
        // .pte runs on the CPU executor only here. Reject other backends so the
        // recorded backend always matches what actually ran.
        require(requestedBackend == Backend.CPU) {
            "ExecuTorchEngine backend $requestedBackend is not implemented yet (CPU only)."
        }
    }

    override val runtime = Runtime.EXECUTORCH
    override val backend = Backend.CPU

    private var module: Module? = null

    // Fixed synthetic input — allocated once, reused every inference
    private val inputData = FloatArray(1 * 3 * 224 * 224)
    private val inputShape = longArrayOf(1, 3, 224, 224)

    override fun loadModel(modelPath: String, precision: Precision): Double {
        val start = SystemClock.elapsedRealtimeNanos()
        module = Module.load(modelPath)

        // Deterministic non-zero input (fixed seed → identical every run)
        val rng = java.util.Random(42L)
        for (i in inputData.indices) inputData[i] = rng.nextFloat()

        runInference()   // first inference included in cold-start
        return (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0
    }

    override fun runInference(): Double {
        val tensor = Tensor.fromBlob(inputData, inputShape)
        val start = SystemClock.elapsedRealtimeNanos()
        module!!.forward(EValue.from(tensor))
        return (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0
    }

    override fun unloadModel() {
        module?.destroy()
        module = null
    }
}
