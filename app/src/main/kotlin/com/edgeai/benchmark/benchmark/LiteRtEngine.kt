package com.edgeai.benchmark.benchmark

import android.content.Context
import android.os.SystemClock
import com.edgeai.benchmark.model.Backend
import com.edgeai.benchmark.model.Precision
import com.edgeai.benchmark.model.Runtime
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Wraps TensorFlow Lite / LiteRT for benchmarking.
 *
 * Supports:
 *   - Backend.CPU   → plain Interpreter, optional XNNPACK threads
 *   - Backend.GPU_DELEGATE → GpuDelegate
 */
class LiteRtEngine(
    context: Context,
    override val backend: Backend = Backend.CPU,
    private val numThreads: Int = 4
) : BenchmarkEngine(context) {

    override val runtime = Runtime.LITERT

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    // Allocated after loadModel() using actual tensor sizes from the interpreter.
    // FP32 model: input=602112 bytes (1×224×224×3×4), INT8 model: input=150528 bytes (1×224×224×3×1)
    private var inputBuffer: ByteBuffer? = null
    private var outputBuffer: ByteBuffer? = null

    override fun loadModel(modelPath: String, precision: Precision): Double {
        val start = SystemClock.elapsedRealtimeNanos()

        val mappedBuffer = loadMappedBuffer(modelPath)

        val options = Interpreter.Options().apply {
            numThreads = this@LiteRtEngine.numThreads
            useXNNPACK = (backend == Backend.CPU)

            if (backend == Backend.GPU_DELEGATE) {
                gpuDelegate = GpuDelegate()
                addDelegate(gpuDelegate!!)
            }
        }

        interpreter = Interpreter(mappedBuffer, options)
        interpreter!!.allocateTensors()

        // Derive buffer sizes from the loaded model's actual tensor specs so that
        // both FP32 (float) and INT8 (uint8) models work without manual size math.
        val interp = interpreter!!
        inputBuffer  = ByteBuffer.allocateDirect(interp.getInputTensor(0).numBytes())
            .order(ByteOrder.nativeOrder())
        outputBuffer = ByteBuffer.allocateDirect(interp.getOutputTensor(0).numBytes())
            .order(ByteOrder.nativeOrder())

        // First inference included in cold-start measurement
        runInference()

        val elapsedMs = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0
        return elapsedMs
    }

    override fun runInference(): Double {
        val interp = requireNotNull(interpreter) { "Model not loaded. Call loadModel() first." }
        inputBuffer!!.rewind()
        outputBuffer!!.rewind()

        val start = SystemClock.elapsedRealtimeNanos()
        interp.run(inputBuffer!!, outputBuffer!!)
        return (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0
    }

    override fun unloadModel() {
        interpreter?.close()
        gpuDelegate?.close()
        interpreter = null
        gpuDelegate = null
        inputBuffer = null
        outputBuffer = null
    }

    // ------------------------------------------------------------------

    private fun loadMappedBuffer(path: String): MappedByteBuffer {
        val file = File(path)
        require(file.exists()) { "Model file not found: $path" }
        return file.inputStream().channel.map(
            FileChannel.MapMode.READ_ONLY, 0, file.length()
        )
    }
}
