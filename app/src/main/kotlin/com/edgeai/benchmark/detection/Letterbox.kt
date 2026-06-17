package com.edgeai.benchmark.detection

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect

/** Resize+pad transform info, used to map model-space boxes back to the original image. */
data class LetterboxInfo(val scale: Float, val padX: Int, val padY: Int)

/**
 * Letterbox: resize keeping aspect ratio, pad to [SIZE]×[SIZE] with gray (114).
 * Matches scripts/eval/yolo_detect_reference.py exactly so detections align.
 */
object Letterbox {
    const val SIZE = 640
    private const val PAD = 114

    fun apply(src: Bitmap): Pair<Bitmap, LetterboxInfo> {
        val w = src.width
        val h = src.height
        val scale = minOf(SIZE.toFloat() / h, SIZE.toFloat() / w)
        val nw = Math.round(w * scale)
        val nh = Math.round(h * scale)
        val padX = (SIZE - nw) / 2
        val padY = (SIZE - nh) / 2

        val out = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(Color.rgb(PAD, PAD, PAD))

        val resized = Bitmap.createScaledBitmap(src, nw, nh, /* filter = */ true)
        canvas.drawBitmap(resized, null, Rect(padX, padY, padX + nw, padY + nh), Paint(Paint.FILTER_BITMAP_FLAG))
        if (resized != src) resized.recycle()

        return out to LetterboxInfo(scale, padX, padY)
    }
}
