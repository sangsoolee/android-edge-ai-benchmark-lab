package com.edgeai.benchmark.benchmark

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.SystemClock
import com.edgeai.benchmark.model.Backend
import com.edgeai.benchmark.model.Precision
import com.edgeai.benchmark.model.Runtime
import java.nio.FloatBuffer
import java.io.File

/**
 * Wraps ONNX Runtime Android for benchmarking.
 *
 * Supports:
 *   - Backend.CPU → plain IntraOpNumThreads session
 *
 * Input layout: NCHW float32 (1×3×224×224), fixed synthetic tensor.
 * ONNX Runtime Android accepts NCHW directly unlike TFLite (NHWC).
 */
class OnnxEngine(
    context: Context,
    requestedBackend: Backend = Backend.CPU,
    private val numThreads: Int = 4
) : BenchmarkEngine(context) {

    init {
        // Only the CPU path is implemented. Reject anything else so the recorded
        // backend can never disagree with what actually ran (NNAPI/GPU are TODO).
        require(requestedBackend == Backend.CPU) {
            "OnnxEngine backend $requestedBackend is not implemented yet (CPU only)."
        }
    }

    override val runtime = Runtime.ONNX_RUNTIME
    override val backend = Backend.CPU

    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    // Fixed synthetic input: 1×3×224×224 float32, allocated once and reused
    private val inputData: FloatArray = FloatArray(1 * 3 * 224 * 224)
    private val inputShape = longArrayOf(1, 3, 224, 224)

    override fun loadModel(modelPath: String, precision: Precision): Double {
        val start = SystemClock.elapsedRealtimeNanos()

        ortEnv = OrtEnvironment.getEnvironment()

        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(numThreads)
            // TODO(v0.2): NNAPI delegate path
            // addNnapi()
        }

        ortSession = ortEnv!!.createSession(
            File(modelPath).readBytes(),
            sessionOptions
        )

        // First inference included in cold-start measurement
        runInference()

        return (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0
    }

    override fun runInference(): Double {
        val env     = requireNotNull(ortEnv)     { "Model not loaded. Call loadModel() first." }
        val session = requireNotNull(ortSession) { "Model not loaded. Call loadModel() first." }

        val inputName = session.inputNames.iterator().next()
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), inputShape)

        val start = SystemClock.elapsedRealtimeNanos()
        tensor.use { t ->
            session.run(mapOf(inputName to t)).use { /* result discarded */ }
        }
        return (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0
    }

    override fun unloadModel() {
        ortSession?.close()
        ortEnv?.close()
        ortSession = null
        ortEnv = null
    }
}
