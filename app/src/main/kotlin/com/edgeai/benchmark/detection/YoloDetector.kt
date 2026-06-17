package com.edgeai.benchmark.detection

import android.graphics.Bitmap
import android.os.SystemClock
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Runs YOLOv8 detection (LiteRT/TFLite) end to end: letterbox → inference →
 * decode → class-aware NMS. Supports FP32 and full-integer INT8/UINT8 models
 * (input quantize / output dequantize from the model's tensor params).
 *
 * Phase boundaries (REVIEW_PLAN_v0.5.3):
 *  - preprocess : letterbox + RGB extract + normalize + (INT8) input quantize
 *  - inference  : interpreter.run only
 *  - postprocess: output read + (INT8) dequant + decode + NMS
 *  - end-to-end : measured by an outer timer over the three
 * Rendering is NOT part of any phase.
 */
class YoloDetector(modelPath: String, numThreads: Int = 4) : AutoCloseable {

    private val interpreter: Interpreter
    private val inputBuffer: ByteBuffer
    private val outputBuffer: ByteBuffer
    private val outputShape: IntArray
    private val size = Letterbox.SIZE

    private val inDtype: DataType
    private val inScale: Float
    private val inZero: Int
    private val outDtype: DataType
    private val outScale: Float
    private val outZero: Int
    private val outCount: Int

    init {
        val options = Interpreter.Options().apply {
            this.numThreads = numThreads
            useXNNPACK = true
        }
        interpreter = Interpreter(loadModel(modelPath), options)
        interpreter.allocateTensors()

        val inT = interpreter.getInputTensor(0)
        inDtype = inT.dataType()
        inT.quantizationParams().let { inScale = it.scale; inZero = it.zeroPoint }
        inputBuffer = ByteBuffer.allocateDirect(inT.numBytes()).order(ByteOrder.nativeOrder())

        val outT = interpreter.getOutputTensor(0)
        outDtype = outT.dataType()
        outputShape = outT.shape()
        outT.quantizationParams().let { outScale = it.scale; outZero = it.zeroPoint }
        outCount = outputShape.fold(1) { a, d -> a * d }
        outputBuffer = ByteBuffer.allocateDirect(outT.numBytes()).order(ByteOrder.nativeOrder())
    }

    /** Visualization path (untimed). */
    fun detect(src: Bitmap): List<Detection> {
        val (padded, lb) = Letterbox.apply(src)
        fillInput(padded)
        if (padded != src) padded.recycle()
        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)
        return Nms.apply(YoloOutputDecoder.decode(readOutput(), outputShape, lb))
    }

    /** One run split into phases (preprocess / inference / postprocess / end-to-end). */
    fun detectTimed(src: Bitmap): PhaseTiming {
        val t0 = SystemClock.elapsedRealtimeNanos()
        val (padded, lb) = Letterbox.apply(src)
        fillInput(padded)
        if (padded != src) padded.recycle()
        val t1 = SystemClock.elapsedRealtimeNanos()

        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)
        val t2 = SystemClock.elapsedRealtimeNanos()

        val dets = Nms.apply(YoloOutputDecoder.decode(readOutput(), outputShape, lb))
        val t3 = SystemClock.elapsedRealtimeNanos()

        return PhaseTiming(
            preprocessMs = (t1 - t0) / 1_000_000.0,
            inferenceMs  = (t2 - t1) / 1_000_000.0,
            postprocessMs = (t3 - t2) / 1_000_000.0,
            endToEndMs   = (t3 - t0) / 1_000_000.0,
            detectionCount = dets.size
        )
    }

    /** NHWC, RGB, normalised to [0,1]; quantized to the input dtype if INT8/UINT8. */
    private fun fillInput(img: Bitmap) {
        inputBuffer.rewind()
        val pixels = IntArray(size * size)
        img.getPixels(pixels, 0, size, 0, 0, size, size)
        val isFloat = inDtype == DataType.FLOAT32
        for (p in pixels) {
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            if (isFloat) {
                inputBuffer.putFloat(r); inputBuffer.putFloat(g); inputBuffer.putFloat(b)
            } else {
                inputBuffer.put(quant(r)); inputBuffer.put(quant(g)); inputBuffer.put(quant(b))
            }
        }
        inputBuffer.rewind()
    }

    /** float [0,1] → quantized byte for the model's input dtype. */
    private fun quant(v: Float): Byte {
        val q = Math.round(v / inScale + inZero)
        return if (inDtype == DataType.UINT8) q.coerceIn(0, 255).toByte()
               else q.coerceIn(-128, 127).toByte()
    }

    /** Read output to FloatArray, dequantizing if the output is INT8/UINT8. */
    private fun readOutput(): FloatArray {
        outputBuffer.rewind()
        val out = FloatArray(outCount)
        when (outDtype) {
            DataType.FLOAT32 -> outputBuffer.asFloatBuffer().get(out)
            DataType.UINT8 -> for (i in 0 until outCount)
                out[i] = ((outputBuffer.get().toInt() and 0xFF) - outZero) * outScale
            else -> for (i in 0 until outCount)  // INT8
                out[i] = (outputBuffer.get().toInt() - outZero) * outScale
        }
        return out
    }

    private fun loadModel(path: String): MappedByteBuffer {
        val file = File(path)
        require(file.exists()) { "Model not found: $path" }
        return file.inputStream().channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
    }

    override fun close() = interpreter.close()
}
