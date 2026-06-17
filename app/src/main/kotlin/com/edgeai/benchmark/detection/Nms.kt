package com.edgeai.benchmark.detection

/**
 * Class-aware non-maximum suppression. Matches the Python reference:
 * sort by score desc, drop a box if it overlaps a kept box of the SAME class
 * with IoU > [IOU_THRES]; cap at [MAX_DET].
 */
object Nms {
    const val IOU_THRES = 0.45f
    const val MAX_DET = 100

    fun apply(dets: List<Detection>): List<Detection> {
        val sorted = dets.sortedByDescending { it.score }
        val kept = ArrayList<Detection>()
        for (d in sorted) {
            val suppressed = kept.any { k -> k.classId == d.classId && iou(k, d) > IOU_THRES }
            if (!suppressed) kept.add(d)
            if (kept.size >= MAX_DET) break
        }
        return kept
    }

    private fun iou(a: Detection, b: Detection): Float {
        val ix1 = maxOf(a.x1, b.x1); val iy1 = maxOf(a.y1, b.y1)
        val ix2 = minOf(a.x2, b.x2); val iy2 = minOf(a.y2, b.y2)
        val iw = maxOf(0f, ix2 - ix1); val ih = maxOf(0f, iy2 - iy1)
        val inter = iw * ih
        val areaA = maxOf(0f, a.x2 - a.x1) * maxOf(0f, a.y2 - a.y1)
        val areaB = maxOf(0f, b.x2 - b.x1) * maxOf(0f, b.y2 - b.y1)
        val union = areaA + areaB - inter
        return if (union > 0f) inter / union else 0f
    }
}
