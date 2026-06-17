package com.edgeai.benchmark.detection

import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Writes detections to JSON in the SAME shape as the Python reference
 * (scripts/eval/yolo_detect_reference.py), so android_<img>.json and
 * python_<img>.json can be diffed directly.
 *
 *   [ { "class_id": 0, "label": "person", "score": 0.873,
 *       "box_xyxy": [123.4, 55.2, 320.1, 480.9] } ]
 */
object DetectionJsonWriter {

    fun write(dets: List<Detection>, file: File): File {
        val arr = JSONArray()
        for (d in dets) {
            val box = JSONArray().apply {
                put(round1(d.x1)); put(round1(d.y1)); put(round1(d.x2)); put(round1(d.y2))
            }
            arr.put(JSONObject().apply {
                put("class_id", d.classId)
                put("label", d.label)
                put("score", round4(d.score))
                put("box_xyxy", box)
            })
        }
        file.parentFile?.mkdirs()
        file.writeText(arr.toString(2))
        return file
    }

    private fun round1(v: Float): Double = Math.round(v * 10.0) / 10.0
    private fun round4(v: Float): Double = Math.round(v * 10000.0) / 10000.0
}
