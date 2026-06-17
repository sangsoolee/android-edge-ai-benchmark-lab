package com.edgeai.benchmark.detection

/**
 * A single detection in ORIGINAL image coordinates.
 * box = [x1, y1, x2, y2] (top-left, bottom-right) in pixels.
 *
 * Mirrors the Python reference JSON (scripts/eval/yolo_detect_reference.py) so
 * android_<img>.json and python_<img>.json can be diffed directly.
 */
data class Detection(
    val classId: Int,
    val label: String,
    val score: Float,
    val box: FloatArray   // x1, y1, x2, y2
) {
    val x1 get() = box[0]
    val y1 get() = box[1]
    val x2 get() = box[2]
    val y2 get() = box[3]
}

/** COCO-80 class names, in YOLOv8 class-index order (matches the Python reference). */
object Coco {
    val NAMES: List<String> = listOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    )

    fun label(classId: Int): String = NAMES.getOrElse(classId) { "cls$classId" }
}
