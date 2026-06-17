package com.edgeai.benchmark.detection

/**
 * Decodes a raw YOLOv8 head output into candidate [Detection]s (before NMS),
 * in ORIGINAL image coordinates. Mirrors scripts/eval/yolo_detect_reference.py.
 *
 * Contract (locked, see REVIEW_PLAN_v0.5.2):
 *  - output normalised to [8400, 84]; row = [cx, cy, w, h, 80 class scores]
 *  - conf = max class score (YOLOv8 has no objectness); class = argmax
 *  - NO anchor-grid reconstruction (boxes are already decoded by the export)
 *  - TFLite emits normalised 0–1 coords (×640); ONNX emits input pixels
 *  - cxcywh → xyxy → letterbox restore to original image
 */
object YoloOutputDecoder {
    const val CONF_THRES = 0.25f
    const val INPUT = Letterbox.SIZE.toFloat()
    private const val NUM_ATTRS = 84
    private const val NUM_CLASSES = 80

    /**
     * @param output flattened model output
     * @param shape  output tensor shape, e.g. [1,84,8400] or [1,8400,84]
     */
    fun decode(output: FloatArray, shape: IntArray, lb: LetterboxInfo): List<Detection> {
        // Identify (numAnchors, attrMajor) from the non-batch dims.
        val d1 = shape[shape.size - 2]
        val d2 = shape[shape.size - 1]
        val attrMajor: Boolean      // true => layout [.., 84, 8400]
        val numAnchors: Int
        when (NUM_ATTRS) {
            d1 -> { attrMajor = true;  numAnchors = d2 }
            d2 -> { attrMajor = false; numAnchors = d1 }
            else -> error("Unexpected YOLO output shape ${shape.joinToString()}")
        }

        fun at(anchor: Int, attr: Int): Float =
            if (attrMajor) output[attr * numAnchors + anchor]
            else output[anchor * NUM_ATTRS + attr]

        // Pass 1: confidence filter, collect cxcywh.
        val cand = ArrayList<FloatArray>()  // [cx,cy,w,h,score,cls]
        for (a in 0 until numAnchors) {
            var bestCls = 0
            var bestScore = at(a, 4)
            for (c in 1 until NUM_CLASSES) {
                val s = at(a, 4 + c)
                if (s > bestScore) { bestScore = s; bestCls = c }
            }
            if (bestScore >= CONF_THRES) {
                cand.add(floatArrayOf(at(a, 0), at(a, 1), at(a, 2), at(a, 3),
                                      bestScore, bestCls.toFloat()))
            }
        }
        if (cand.isEmpty()) return emptyList()

        // Coordinate scale: normalised (≤~1.5) → input pixels.
        var maxCoord = 0f
        for (c in cand) maxCoord = maxOf(maxCoord, c[0], c[1], c[2], c[3])
        val sf = if (maxCoord <= 1.5f) INPUT else 1f

        val out = ArrayList<Detection>(cand.size)
        for (c in cand) {
            val cx = c[0] * sf; val cy = c[1] * sf; val w = c[2] * sf; val h = c[3] * sf
            // cxcywh -> xyxy (input/640 space) -> letterbox restore to original
            val x1 = (cx - w / 2f - lb.padX) / lb.scale
            val y1 = (cy - h / 2f - lb.padY) / lb.scale
            val x2 = (cx + w / 2f - lb.padX) / lb.scale
            val y2 = (cy + h / 2f - lb.padY) / lb.scale
            val cls = c[5].toInt()
            out.add(Detection(cls, Coco.label(cls), c[4], floatArrayOf(x1, y1, x2, y2)))
        }
        return out
    }
}
