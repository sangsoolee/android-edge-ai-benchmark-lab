#!/usr/bin/env python3
"""
export_tflite.py
----------------
Exports a PyTorch vision model to TFLite / LiteRT format.

Pipeline:
  PyTorch → ONNX → TF SavedModel (onnx2tf) → TFLite (TFLiteConverter)

Supported precisions:
  fp32  → standard float32 TFLite flatbuffer
  fp16  → float16 quantization (weights only)
  int8  → full integer quantization with representative dataset

Usage:
  python export_tflite.py --model mobilenet_v3_small --precision fp32
  python export_tflite.py --model mobilenet_v3_small --precision int8 --calib-samples 200

Output:
  ../../models/<model>_<precision>.tflite
"""

import argparse
import shutil
import numpy as np
import sys
import time
from pathlib import Path

import torch
import torchvision.models as tv_models
import tensorflow as tf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = {
    "mobilenet_v3_small": lambda: tv_models.mobilenet_v3_small(weights="IMAGENET1K_V1"),
    "mobilenet_v3_large": lambda: tv_models.mobilenet_v3_large(weights="IMAGENET1K_V1"),
    "efficientnet_b0":    lambda: tv_models.efficientnet_b0(weights="IMAGENET1K_V1"),
}

INPUT_SHAPE = (1, 3, 224, 224)   # NCHW
OUTPUT_DIR  = Path(__file__).parent.parent.parent / "models"

# ---------------------------------------------------------------------------
# Stage 1: PyTorch → ONNX
# ---------------------------------------------------------------------------

def export_to_onnx(model: torch.nn.Module, model_name: str) -> Path:
    onnx_path = OUTPUT_DIR / f"{model_name}_tmp.onnx"
    dummy = torch.randn(*INPUT_SHAPE)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,   # force legacy exporter; dynamo path has heavier onnxscript deps
    )
    print(f"  [ONNX] saved → {onnx_path}")
    return onnx_path


# ---------------------------------------------------------------------------
# Stage 2: ONNX → TF SavedModel
# ---------------------------------------------------------------------------

def onnx_to_saved_model(onnx_path: Path, saved_model_dir: Path) -> None:
    """Convert ONNX to TF SavedModel via onnx2tf.
    TFLiteConverter.from_saved_model() requires a SavedModel directory,
    not a .tflite file — this stage produces that intermediate artifact.
    """
    try:
        import onnx2tf
    except ImportError:
        sys.exit("Install onnx2tf: pip install onnx2tf")

    saved_model_dir.mkdir(parents=True, exist_ok=True)
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        output_tfv1_pb=False,
        non_verbose=True,
    )
    print(f"  [SavedModel] saved → {saved_model_dir}")


# ---------------------------------------------------------------------------
# Stage 3: TF SavedModel → TFLite (per precision)
# ---------------------------------------------------------------------------

def to_tflite_fp32(saved_model_dir: Path, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    output_path.write_bytes(converter.convert())
    print(f"  [TFLite FP32] saved → {output_path}")


def to_tflite_fp16(saved_model_dir: Path, output_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    output_path.write_bytes(converter.convert())
    print(f"  [TFLite FP16] saved → {output_path}")


def to_tflite_int8(saved_model_dir: Path, output_path: Path, calib_samples: int) -> None:
    def representative_dataset():
        # NHWC layout expected by TFLiteConverter
        for _ in range(calib_samples):
            yield [np.random.randn(1, 224, 224, 3).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.uint8
    converter.inference_output_type = tf.uint8
    output_path.write_bytes(converter.convert())
    print(f"  [TFLite INT8] saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model → TFLite")
    parser.add_argument("--model",         required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--precision",     required=True, choices=["fp32", "fp16", "int8"])
    parser.add_argument("--calib-samples", type=int, default=100,
                        help="Calibration samples for INT8 quantization (default: 100)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model:     {args.model}")
    print(f"  Precision: {args.precision}")
    print(f"  Output:    {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()

    # 1. PyTorch → ONNX
    print("[1/3] Loading PyTorch model and exporting to ONNX...")
    pt_model = SUPPORTED_MODELS[args.model]()
    pt_model.eval()
    onnx_path = export_to_onnx(pt_model, args.model)

    # 2. ONNX → TF SavedModel
    print("[2/3] Converting ONNX → TF SavedModel...")
    saved_model_dir = OUTPUT_DIR / f"{args.model}_saved_model"
    onnx_to_saved_model(onnx_path, saved_model_dir)

    # 3. SavedModel → TFLite
    print(f"[3/3] Converting SavedModel → TFLite ({args.precision})...")
    out_path = OUTPUT_DIR / f"{args.model}_{args.precision}.tflite"

    if args.precision == "fp32":
        to_tflite_fp32(saved_model_dir, out_path)
    elif args.precision == "fp16":
        to_tflite_fp16(saved_model_dir, out_path)
    elif args.precision == "int8":
        to_tflite_int8(saved_model_dir, out_path, args.calib_samples)

    # Cleanup intermediate files
    onnx_path.unlink(missing_ok=True)
    shutil.rmtree(saved_model_dir, ignore_errors=True)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    elapsed = time.perf_counter() - t0

    print(f"\n✅  Done in {elapsed:.1f}s")
    print(f"   File:    {out_path.name}")
    print(f"   Size:    {size_mb:.2f} MB\n")


if __name__ == "__main__":
    main()
