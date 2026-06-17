package com.edgeai.benchmark.detection

import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Runs YOLOv8 detection (LiteRT/TFLite) end to end: letterbox → inference →
 * decode → NMS. FP32 model only for v0.5.2 (INT8 detection on device is a
 * follow-up — needs input quantization to match the model's scale/zero-point).
 */
class YoloDetector(modelPath: String, numThreads: Int = 4) : AutoCloseable {

    private val interpreter: Interpreter
    private val inputBuffer: ByteBuffer
    private val outputBuffer: ByteBuffer
    private val outputShape: IntArray
    private val size = Letterbox.SIZE

    init {
        val options = Interpreter.Options().apply {
            this.numThreads = numThreads
            useXNNPACK = true
        }
        interpreter = Interpreter(loadModel(modelPath), options)
        interpreter.allocateTensors()

        val inT = interpreter.getInputTensor(0)
        require(inT.dataType() == org.tensorflow.lite.DataType.FLOAT32) {
            "YoloDetector supports FP32 models only (got ${inT.dataType()}). INT8 detection is a v0.5.x follow-up."
        }
        inputBuffer = ByteBuffer.allocateDirect(inT.numBytes()).order(ByteOrder.nativeOrder())

        val outT = interpreter.getOutputTensor(0)
        outputShape = outT.shape()
        outputBuffer = ByteBuffer.allocateDirect(outT.numBytes()).order(ByteOrder.nativeOrder())
    }

    fun detect(src: Bitmap): List<Detection> {
        val (padded, lb) = Letterbox.apply(src)
        fillInput(padded)
        if (padded != src) padded.recycle()

        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)

        outputBuffer.rewind()
        val n = outputShape.fold(1) { acc, d -> acc * d }
        val out = FloatArray(n)
        outputBuffer.asFloatBuffer().get(out)

        val candidates = YoloOutputDecoder.decode(out, outputShape, lb)
        return Nms.apply(candidates)
    }

    /** NHWC, RGB, normalised to [0,1] — matches the Python reference preprocessing. */
    private fun fillInput(img: Bitmap) {
        inputBuffer.rewind()
        val pixels = IntArray(size * size)
        img.getPixels(pixels, 0, size, 0, 0, size, size)
        for (p in pixels) {
            inputBuffer.putFloat(((p shr 16) and 0xFF) / 255f)  // R
            inputBuffer.putFloat(((p shr 8) and 0xFF) / 255f)   // G
            inputBuffer.putFloat((p and 0xFF) / 255f)           // B
        }
        inputBuffer.rewind()
    }

    private fun loadModel(path: String): MappedByteBuffer {
        val file = File(path)
        require(file.exists()) { "Model not found: $path" }
        return file.inputStream().channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
    }

    override fun close() = interpreter.close()
}
