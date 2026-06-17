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

  # INT8 with real ImageNet calibration data (recommended — avoids quantization bias)
  python export_tflite.py --model mobilenet_v3_small --precision int8 \\
    --representative-data /path/to/imagenet/val \\
    --representative-samples 500

Output:
  ../../models/<model>_<precision>.tflite
"""

import argparse
import glob
import random
import shutil
import numpy as np
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torchvision.models as tv_models
import tensorflow as tf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = {
    "mobilenet_v3_small": lambda: tv_models.mobilenet_v3_small(weights="IMAGENET1K_V1"),
    "mobilenet_v3_large": lambda: tv_models.mobilenet_v3_large(weights="IMAGENET1K_V1"),
    # MobileNetV2: ReLU6 activations (no hard-swish/SE) → quantizes cleanly under
    # full-integer PTQ, unlike MobileNetV3 which collapses. Used for the v0.4
    # FP32-vs-INT8 accuracy comparison.
    "mobilenet_v2":       lambda: tv_models.mobilenet_v2(weights="IMAGENET1K_V1"),
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


def _preprocess_image(image_path: str) -> np.ndarray:
    """Load one image → float32 NHWC (1, 224, 224, 3), torchvision-normalized."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return ((arr - mean) / std)[np.newaxis]   # (1, 224, 224, 3)


def _make_real_dataset(data_dir: str, n_samples: int):
    """Calibration generator using real ImageNet validation images."""
    patterns = [f"{data_dir}/**/*.JPEG", f"{data_dir}/**/*.jpg", f"{data_dir}/**/*.jpeg"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    if not files:
        raise ValueError(f"No images found in {data_dir}. Provide --representative-data correctly.")
    random.shuffle(files)
    chosen = files[:n_samples]
    print(f"  Found {len(files)} images, using {len(chosen)} for calibration")

    def generator():
        for path in chosen:
            try:
                yield [_preprocess_image(path)]
            except Exception:
                pass  # skip unreadable images

    return generator


def to_tflite_int8(
    saved_model_dir: Path,
    output_path: Path,
    calib_samples: int,
    representative_data_dir: Optional[str] = None,
) -> None:
    if representative_data_dir:
        representative_dataset = _make_real_dataset(representative_data_dir, calib_samples)
        print(f"  Using real ImageNet images from: {representative_data_dir}")
    else:
        def representative_dataset():
            # NHWC layout expected by TFLiteConverter; random noise approximates
            # normalized distribution but real images give better calibration
            for _ in range(calib_samples):
                yield [np.random.randn(1, 224, 224, 3).astype(np.float32)]
        print(f"  Using random noise calibration ({calib_samples} samples)")
        print("  Tip: pass --representative-data for better accuracy on real images")

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
    parser.add_argument("--model",     required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--precision", required=True, choices=["fp32", "fp16", "int8"])
    parser.add_argument("--calib-samples", type=int, default=100,
                        help="Calibration samples for INT8 quantization (default: 100)")
    parser.add_argument("--representative-data", type=str, default=None,
                        help="Directory of real ImageNet val images for INT8 calibration "
                             "(recommended over random noise). Supports flat or nested JPEG trees.")
    parser.add_argument("--representative-samples", type=int, default=500,
                        help="How many images to use from --representative-data (default: 500)")
    args = parser.parse_args()

    # --representative-samples overrides --calib-samples when real data is provided
    if args.representative_data:
        args.calib_samples = args.representative_samples

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
        to_tflite_int8(saved_model_dir, out_path, args.calib_samples,
                       representative_data_dir=args.representative_data)

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
