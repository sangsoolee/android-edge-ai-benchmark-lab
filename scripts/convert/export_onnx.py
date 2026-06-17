#!/usr/bin/env python3
"""
export_onnx.py
--------------
Exports a PyTorch vision model to ONNX format for use with ONNX Runtime Android.

Supported precisions:
  fp32  → standard float32 ONNX model
  fp16  → float16 via onnxconverter-common
  int8  → dynamic quantization via onnxruntime.quantization

Usage:
  python export_onnx.py --model mobilenet_v3_small --precision fp32
  python export_onnx.py --model mobilenet_v3_small --precision int8

Output:
  ../../models/<model>_<precision>.onnx
"""

import argparse
import time
from pathlib import Path

import torch
import torchvision.models as tv_models

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = {
    "mobilenet_v3_small": lambda: tv_models.mobilenet_v3_small(weights="IMAGENET1K_V1"),
    "mobilenet_v3_large": lambda: tv_models.mobilenet_v3_large(weights="IMAGENET1K_V1"),
    "efficientnet_b0":    lambda: tv_models.efficientnet_b0(weights="IMAGENET1K_V1"),
}

INPUT_SHAPE = (1, 3, 224, 224)
OUTPUT_DIR  = Path(__file__).parent.parent.parent / "models"

# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_fp32(model: torch.nn.Module, model_name: str) -> Path:
    out = OUTPUT_DIR / f"{model_name}_fp32.onnx"
    dummy = torch.randn(*INPUT_SHAPE)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,   # force legacy exporter; dynamo path separates weights into external files
    )
    print(f"  [ONNX FP32] → {out}")
    return out


def export_fp16(fp32_path: Path, model_name: str) -> Path:
    try:
        from onnxconverter_common import float16
        import onnx
    except ImportError:
        raise SystemExit("pip install onnxconverter-common onnx")

    out = OUTPUT_DIR / f"{model_name}_fp16.onnx"
    model_fp32 = onnx.load(str(fp32_path))
    model_fp16 = float16.convert_float_to_float16(model_fp32)
    onnx.save(model_fp16, str(out))
    print(f"  [ONNX FP16] → {out}")
    return out


def export_int8(fp32_path: Path, model_name: str) -> Path:
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        raise SystemExit("pip install onnxruntime")

    out = OUTPUT_DIR / f"{model_name}_int8.onnx"
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(out),
        weight_type=QuantType.QInt8,
    )
    print(f"  [ONNX INT8]  → {out}")
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(path: Path) -> None:
    """Quick shape-check with onnxruntime CPU."""
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        out = sess.run(None, {inp_name: dummy})
        print(f"  ✅ Validation OK — output shape: {out[0].shape}")
    except Exception as e:
        print(f"  ⚠️  Validation failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model → ONNX")
    parser.add_argument("--model",     required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--precision", required=True, choices=["fp32", "fp16", "int8"])
    parser.add_argument("--validate",  action="store_true", help="Run CPU inference check after export")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model:     {args.model}")
    print(f"  Precision: {args.precision}")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()

    print("[1/2] Loading PyTorch model...")
    model = SUPPORTED_MODELS[args.model]()
    model.eval()

    print("[2/2] Exporting...")
    fp32_path = OUTPUT_DIR / f"{args.model}_fp32.onnx"

    if args.precision == "fp32":
        out_path = export_fp32(model, args.model)
    elif args.precision == "fp16":
        if not fp32_path.exists():
            fp32_path = export_fp32(model, args.model)
        out_path = export_fp16(fp32_path, args.model)
    elif args.precision == "int8":
        if not fp32_path.exists():
            fp32_path = export_fp32(model, args.model)
        out_path = export_int8(fp32_path, args.model)

    if args.validate:
        print("\nValidating...")
        validate(out_path)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n✅  Done in {time.perf_counter() - t0:.1f}s")
    print(f"   File:  {out_path.name}")
    print(f"   Size:  {size_mb:.2f} MB\n")


if __name__ == "__main__":
    main()
