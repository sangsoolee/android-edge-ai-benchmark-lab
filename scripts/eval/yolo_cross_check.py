#!/usr/bin/env python3
"""
yolo_cross_check.py
-------------------
Validates that the YOLOv8 ONNX and TFLite exports compute the same detector,
by feeding one fixed-seed 640x640 input and comparing the raw output tensors.
v0.5 object-detection correctness check (analogue of cross_runtime_check.py).

YOLOv8 output is (1, 84, 8400): 84 = 4 bbox (xywh) + 80 class scores, over 8400
anchors. ONNX takes NCHW (1,3,640,640); TFLite takes NHWC (1,640,640,3) and may
emit the output transposed — both are normalised to (8400, 84) before comparing.

Requirements: numpy, ai-edge-litert, onnxruntime  (FP32 models only)

Usage:
  python scripts/eval/yolo_cross_check.py \\
    --tflite models/yolov8n_fp32.tflite \\
    --onnx   models/yolov8n_fp32.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np

SEED = 42
IMG  = 640
NUM_ATTRS = 84  # 4 bbox + 80 COCO classes


def make_inputs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    x_nchw = rng.random((1, 3, IMG, IMG), dtype=np.float32)
    x_nhwc = np.transpose(x_nchw, (0, 2, 3, 1)).copy()
    return x_nchw, x_nhwc


def _to_anchors_major(out: np.ndarray) -> np.ndarray:
    """Normalise a YOLOv8 head output to (num_anchors, 84)."""
    a = np.squeeze(out)                 # drop batch -> (84, 8400) or (8400, 84)
    if a.ndim != 2:
        raise SystemExit(f"Unexpected YOLO output shape {out.shape}")
    if a.shape[0] == NUM_ATTRS:         # (84, 8400) -> (8400, 84)
        a = a.T
    return a.astype(np.float32)


def run_onnx(path: Path, x_nchw: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: x_nchw})[0]
    return _to_anchors_major(out)


def run_tflite(path: Path, x_nhwc: np.ndarray) -> np.ndarray:
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(path)); it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    if inp["dtype"] != np.float32:
        raise SystemExit(f"{path.name}: expected FP32 input, got {inp['dtype']}")
    it.set_tensor(inp["index"], x_nhwc)
    it.invoke()
    return _to_anchors_major(it.get_tensor(outp["index"]))


def top_detection(a: np.ndarray) -> tuple[int, int, float]:
    """Return (anchor_idx, class_id, confidence) of the highest class score."""
    cls = a[:, 4:]                       # (8400, 80)
    flat = int(cls.argmax())
    anchor, class_id = divmod(flat, cls.shape[1])
    return anchor, class_id, float(cls[anchor, class_id])


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 ONNX vs TFLite output check")
    parser.add_argument("--tflite", type=Path, required=True)
    parser.add_argument("--onnx",   type=Path, required=True)
    parser.add_argument("--min-cosine", type=float, default=0.99)
    args = parser.parse_args()

    x_nchw, x_nhwc = make_inputs()

    a_onnx = run_onnx(args.onnx, x_nchw)
    a_tfl  = run_tflite(args.tflite, x_nhwc)
    print(f"  onnx   output → {a_onnx.shape}")
    print(f"  tflite output → {a_tfl.shape}")

    if a_onnx.shape != a_tfl.shape:
        raise SystemExit(f"Shape mismatch after normalisation: {a_onnx.shape} vs {a_tfl.shape}")

    # Class-score block drives the comparison (bbox xywh scaling can differ by export).
    co, ct = a_onnx[:, 4:].ravel(), a_tfl[:, 4:].ravel()
    cosine = float(np.dot(co, ct) / (np.linalg.norm(co) * np.linalg.norm(ct) + 1e-12))
    max_abs = float(np.max(np.abs(a_onnx - a_tfl)))

    o_anchor, o_cls, o_conf = top_detection(a_onnx)
    t_anchor, t_cls, t_conf = top_detection(a_tfl)

    print(f"\n  class-score cosine : {cosine:.5f}")
    print(f"  max abs diff       : {max_abs:.5f}")
    print(f"  onnx   top: class={o_cls} conf={o_conf:.4f} anchor={o_anchor}")
    print(f"  tflite top: class={t_cls} conf={t_conf:.4f} anchor={t_anchor}")
    print(f"  top-class match    : {o_cls == t_cls}\n")

    if cosine < args.min_cosine or o_cls != t_cls:
        print(f"❌ ONNX and TFLite disagree (cosine {cosine:.5f}, "
              f"top-class match {o_cls == t_cls}).")
        sys.exit(1)
    print(f"✅ YOLOv8 ONNX and TFLite agree (cosine {cosine:.5f} ≥ {args.min_cosine}).")


if __name__ == "__main__":
    main()
