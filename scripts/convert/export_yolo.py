#!/usr/bin/env python3
"""
export_yolo.py
--------------
Exports a YOLO detection model (Ultralytics) to ONNX and TFLite for the
on-device benchmark. v0.5 object-detection track.

Ultralytics writes exports next to the weights / in a saved_model dir with its
own naming; this script relocates them to models/<model>_<precision>.<ext> so
they match the rest of the pipeline.

Install:
  pip install ultralytics

Usage:
  # FP32 ONNX + TFLite
  python scripts/convert/export_yolo.py --model yolov8n --precision fp32

  # INT8 TFLite (full-integer) with a calibration dataset yaml (e.g. coco8.yaml)
  python scripts/convert/export_yolo.py --model yolov8n --precision int8 --data coco8.yaml

Output:
  models/yolov8n_fp32.onnx
  models/yolov8n_fp32.tflite
  models/yolov8n_int8.tflite   (with --precision int8)
"""

import argparse
import shutil
import time
from pathlib import Path

SUPPORTED_MODELS = {"yolov8n", "yolov11n", "yolov8s"}  # any ultralytics .pt stem
INPUT_SIZE = 640
OUTPUT_DIR = Path(__file__).parent.parent.parent / "models"


def _find_and_move(produced_glob: str, dest: Path) -> Path:
    """Ultralytics output paths vary by version; glob for the produced artifact."""
    here = Path.cwd()
    candidates = sorted(here.rglob(produced_glob), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"Export produced no file matching '{produced_glob}'. "
                         "Check the ultralytics export logs above.")
    src = candidates[-1]  # most recent
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    print(f"  moved {src.name} -> {dest}")
    return dest


def export_onnx(model_name: str) -> Path:
    from ultralytics import YOLO
    model = YOLO(f"{model_name}.pt")
    print("  [ONNX] exporting ...")
    model.export(format="onnx", imgsz=INPUT_SIZE, opset=13, simplify=True)
    return _find_and_move(f"{model_name}.onnx", OUTPUT_DIR / f"{model_name}_fp32.onnx")


def export_tflite(model_name: str, precision: str, data: str | None) -> Path:
    from ultralytics import YOLO
    model = YOLO(f"{model_name}.pt")

    if precision == "int8":
        if not data:
            raise SystemExit("--precision int8 needs --data (e.g. coco8.yaml) for calibration.")
        print(f"  [TFLite INT8] exporting with calibration data={data} ...")
        model.export(format="tflite", imgsz=INPUT_SIZE, int8=True, data=data)
        produced = f"{model_name}_full_integer_quant.tflite"
        dest = OUTPUT_DIR / f"{model_name}_int8.tflite"
    else:
        print("  [TFLite FP32] exporting ...")
        model.export(format="tflite", imgsz=INPUT_SIZE)
        produced = f"{model_name}_float32.tflite"
        dest = OUTPUT_DIR / f"{model_name}_fp32.tflite"

    return _find_and_move(produced, dest)


def validate_onnx(path: Path) -> None:
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        x = np.random.rand(1, 3, INPUT_SIZE, INPUT_SIZE).astype("float32")
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        print(f"  ✅ ONNX OK — output shape {out.shape}")  # expect (1, 84, 8400)
    except Exception as e:
        print(f"  ⚠️  ONNX validation skipped: {e}")


def validate_tflite(path: Path) -> None:
    try:
        from ai_edge_litert.interpreter import Interpreter
        import numpy as np
        it = Interpreter(model_path=str(path)); it.allocate_tensors()
        inp, outp = it.get_input_details()[0], it.get_output_details()[0]
        if inp["dtype"] == np.float32:
            x = np.random.rand(*inp["shape"]).astype(np.float32)
        else:
            info = np.iinfo(inp["dtype"])   # int8 -> [-128,127], uint8 -> [0,255]
            x = np.random.randint(info.min, info.max + 1, size=inp["shape"], dtype=inp["dtype"])
        it.set_tensor(inp["index"], x); it.invoke()
        print(f"  ✅ TFLite OK — in {inp['shape']} {inp['dtype'].__name__}, "
              f"out {it.get_tensor(outp['index']).shape}")
    except Exception as e:
        print(f"  ⚠️  TFLite validation skipped: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a YOLO model to ONNX + TFLite")
    parser.add_argument("--model", default="yolov8n",
                        help="Ultralytics model stem (yolov8n, yolov11n, ...)")
    parser.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    parser.add_argument("--data", default=None,
                        help="Calibration dataset yaml for INT8 (e.g. coco8.yaml)")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n  Model: {args.model}  Precision: {args.precision}\n{'='*60}\n")
    t0 = time.perf_counter()

    if args.precision == "fp32":
        onnx_path = export_onnx(args.model)
        tflite_path = export_tflite(args.model, "fp32", None)
        if args.validate:
            validate_onnx(onnx_path)
            validate_tflite(tflite_path)
    else:
        tflite_path = export_tflite(args.model, "int8", args.data)
        if args.validate:
            validate_tflite(tflite_path)

    print(f"\n✅  Done in {time.perf_counter() - t0:.1f}s → {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
