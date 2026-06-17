package com.edgeai.benchmark.detection

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint

/** Draws detection boxes + labels onto a copy of the original image. */
object DetectionRenderer {

    fun render(src: Bitmap, dets: List<Detection>): Bitmap {
        val out = src.copy(Bitmap.Config.ARGB_8888, /* mutable = */ true)
        val canvas = Canvas(out)

        val stroke = (maxOf(src.width, src.height) / 200f).coerceAtLeast(2f)
        val box = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = stroke
            color = Color.rgb(255, 80, 80)
        }
        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = (maxOf(src.width, src.height) / 45f).coerceAtLeast(18f)
        }
        val textBg = Paint().apply { color = Color.rgb(255, 80, 80) }

        for (d in dets) {
            canvas.drawRect(d.x1, d.y1, d.x2, d.y2, box)
            val tag = "${d.label} ${"%.2f".format(d.score)}"
            val tw = textPaint.measureText(tag)
            val th = textPaint.textSize
            val ty = (d.y1 - th).coerceAtLeast(0f)
            canvas.drawRect(d.x1, ty, d.x1 + tw + 8f, ty + th + 6f, textBg)
            canvas.drawText(tag, d.x1 + 4f, ty + th, textPaint)
        }
        return out
    }
}
