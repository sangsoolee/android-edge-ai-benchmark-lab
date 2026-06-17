#!/usr/bin/env python3
"""
yolo_detect_reference.py
------------------------
Reference YOLOv8 postprocess (Python) — the acceptance baseline that the Android
implementation (v0.5.2) must match. Implements the locked contract:
  raw output -> [8400, 84] -> (cx,cy,w,h + 80 class scores)
  conf = max(class scores)  (YOLOv8 has no objectness)
  cxcywh -> xyxy, letterbox restore -> original image coords
  class-aware NMS (IoU 0.45), conf 0.25, max 100 detections.

No anchor-grid reconstruction — YOLOv8 export already emits decoded cx,cy,w,h.

Requirements: numpy, pillow, ai-edge-litert (TFLite) and/or onnxruntime (ONNX)

Usage:
  python scripts/eval/yolo_detect_reference.py \\
    --model models/yolov8n_fp32.tflite \\
    --image path/to/sample.jpg \\
    --out-json   results/detection/debug/python_sample.json \\
    --out-image  results/detection/debug/python_sample.png
"""

import argparse
import json
from pathlib import Path

import numpy as np

INPUT_SIZE = 640
CONF_THRES = 0.25
IOU_THRES  = 0.45
MAX_DET    = 100

COCO80 = [
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
    "hair drier", "toothbrush",
]


# ---------------------------------------------------------------------------
# Preprocess (letterbox)
# ---------------------------------------------------------------------------

def letterbox(img: np.ndarray, size: int = INPUT_SIZE):
    """Resize keeping aspect ratio, pad to size×size (gray 114). HWC uint8 in/out.
    Returns (padded, scale, pad_x, pad_y) so boxes can be restored to original."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    from PIL import Image
    resized = np.array(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    out[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return out, scale, pad_x, pad_y


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_model(model_path: Path, padded: np.ndarray) -> np.ndarray:
    """Run YOLOv8 on a 640×640 letterboxed image. Returns raw output as-is."""
    x = padded.astype(np.float32) / 255.0  # HWC [0,1]

    if model_path.suffix == ".onnx":
        import onnxruntime as ort
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        nchw = np.transpose(x, (2, 0, 1))[np.newaxis]  # (1,3,640,640)
        return sess.run(None, {sess.get_inputs()[0].name: nchw})[0]

    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(model_path)); it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    nhwc = x[np.newaxis]  # (1,640,640,3)
    if inp["dtype"] != np.float32:  # INT8: quantize with model params
        scale, zp = inp["quantization"]
        nhwc = np.clip(np.round(nhwc / scale + zp),
                       np.iinfo(inp["dtype"]).min, np.iinfo(inp["dtype"]).max
                       ).astype(inp["dtype"])
    it.set_tensor(inp["index"], nhwc); it.invoke()
    out = it.get_tensor(outp["index"])
    if outp["dtype"] != np.float32:  # dequantize INT8 output
        scale, zp = outp["quantization"]
        out = (out.astype(np.float32) - zp) * scale
    return out


# ---------------------------------------------------------------------------
# Postprocess (locked contract)
# ---------------------------------------------------------------------------

def to_anchors_major(out: np.ndarray) -> np.ndarray:
    a = np.squeeze(out)
    if a.shape[0] == 84:   # (84, 8400) -> (8400, 84)
        a = a.T
    return a.astype(np.float32)


def decode(a: np.ndarray, scale: float, pad_x: int, pad_y: int):
    """a: (8400, 84). Returns list of (x1,y1,x2,y2, score, class_id) in orig coords."""
    boxes = a[:, :4].copy()
    cls_scores = a[:, 4:]
    conf = cls_scores.max(axis=1)
    cls = cls_scores.argmax(axis=1)

    keep = conf >= CONF_THRES
    boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
    if boxes.shape[0] == 0:
        return []

    # Coordinate scale: TFLite emits normalized 0–1; ONNX emits input pixels.
    if boxes[:, :4].max() <= 1.5:
        boxes *= INPUT_SIZE

    # cxcywh -> xyxy (input/640 space)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2; y1 = cy - h / 2; x2 = cx + w / 2; y2 = cy + h / 2

    # Letterbox restore -> original image coords
    x1 = (x1 - pad_x) / scale; x2 = (x2 - pad_x) / scale
    y1 = (y1 - pad_y) / scale; y2 = (y2 - pad_y) / scale

    return list(zip(x1, y1, x2, y2, conf, cls))


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def class_aware_nms(dets: list) -> list:
    """dets: (x1,y1,x2,y2,score,cls). class-aware NMS, IoU 0.45, cap MAX_DET."""
    dets = sorted(dets, key=lambda d: d[4], reverse=True)
    kept = []
    for d in dets:
        if all(not (int(k[5]) == int(d[5]) and iou(k, d) > IOU_THRES) for k in kept):
            kept.append(d)
        if len(kept) >= MAX_DET:
            break
    return kept


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def to_json(dets: list) -> list:
    return [
        {"class_id": int(c), "label": COCO80[int(c)], "score": round(float(s), 4),
         "box_xyxy": [round(float(x1), 1), round(float(y1), 1),
                      round(float(x2), 1), round(float(y2), 1)]}
        for (x1, y1, x2, y2, s, c) in dets
    ]


def draw(img: np.ndarray, dets: list, out_path: Path) -> None:
    from PIL import Image, ImageDraw
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    for (x1, y1, x2, y2, s, c) in dets:
        d.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=3)
        d.text((x1 + 2, max(0, y1 - 11)), f"{COCO80[int(c)]} {s:.2f}", fill=(255, 80, 80))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def main() -> None:
    repo = Path(__file__).parents[2]
    p = argparse.ArgumentParser(description="Reference YOLOv8 detection (Python baseline)")
    p.add_argument("--model", required=True, type=Path, help=".tflite or .onnx")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--out-json", type=Path,
                   default=repo / "results" / "detection" / "debug" / "python_sample.json")
    p.add_argument("--out-image", type=Path,
                   default=repo / "results" / "detection" / "debug" / "python_sample.png")
    args = p.parse_args()

    from PIL import Image
    img = np.array(Image.open(args.image).convert("RGB"))
    padded, scale, pad_x, pad_y = letterbox(img)

    raw = run_model(args.model, padded)
    a = to_anchors_major(raw)
    dets = class_aware_nms(decode(a, scale, pad_x, pad_y))

    js = to_json(dets)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(js, indent=2))
    draw(img, dets, args.out_image)

    print(f"  detections: {len(js)}")
    for d in js[:10]:
        print(f"    {d['label']:14s} {d['score']:.3f}  {d['box_xyxy']}")
    print(f"\n  → {args.out_json}\n  → {args.out_image}")


if __name__ == "__main__":
    main()
