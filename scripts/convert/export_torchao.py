#!/usr/bin/env python3
"""
export_torchao.py
-----------------
Quantizes a PyTorch model using TorchAO (int8 weight-only) and exports to ONNX.

TorchAO vs onnxruntime.quantization.quantize_dynamic:
  - torchao int8_weight_only: quantizes Linear weight tensors to INT8,
    activations stay FP32. Runs quantized matmuls during inference.
  - onnxruntime quantize_dynamic: dynamic quantization applied post-export.
  Both are "dynamic" (no calibration data needed), but TorchAO operates
  at the PyTorch level, giving the optimizer more context.

Requirements:
  pip install torchao  (already in requirements.txt)

Usage:
  python export_torchao.py --model mobilenet_v3_small

Output:
  ../../models/<model>_torchao_int8.onnx
"""

import argparse
import time
from pathlib import Path

import torch
import torchvision.models as tv_models

SUPPORTED_MODELS = {
    "mobilenet_v3_small": lambda: tv_models.mobilenet_v3_small(weights="IMAGENET1K_V1"),
    "efficientnet_b0":    lambda: tv_models.efficientnet_b0(weights="IMAGENET1K_V1"),
}

INPUT_SHAPE = (1, 3, 224, 224)
OUTPUT_DIR  = Path(__file__).parent.parent.parent / "models"

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_torchao_int8(model: torch.nn.Module, model_name: str) -> Path:
    try:
        from torchao.quantization import quantize_, int8_weight_only
    except ImportError:
        raise SystemExit("pip install torchao")

    out = OUTPUT_DIR / f"{model_name}_torchao_int8.onnx"
    dummy = torch.randn(*INPUT_SHAPE)

    print("  [1/3] Applying TorchAO int8_weight_only quantization...")
    quantize_(model, int8_weight_only())
    model.eval()

    print("  [2/3] Exporting quantized model to ONNX...")
    # Unwrap for ONNX export (TorchAO-quantized models need torch.compile or
    # direct export with the weights already quantized)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(out),
            opset_version=17,  # INT8 ops need opset 17+
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            dynamo=False,
        )

    print(f"  [TorchAO INT8] → {out}")
    return out


def validate(path: Path) -> None:
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        out = sess.run(None, {sess.get_inputs()[0].name: dummy})
        print(f"  ✅ Validation OK — output shape: {out[0].shape}")
    except Exception as e:
        print(f"  ⚠️  Validation failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TorchAO INT8 quantize → ONNX export")
    parser.add_argument("--model",    required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model:     {args.model}")
    print(f"  Method:    TorchAO int8_weight_only")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()

    print("[1/2] Loading PyTorch model...")
    model = SUPPORTED_MODELS[args.model]()
    model.eval()

    print("[2/2] Quantizing and exporting...")
    out_path = export_torchao_int8(model, args.model)

    if args.validate:
        print("\nValidating...")
        validate(out_path)

    fp32_path = OUTPUT_DIR / f"{args.model}_fp32.onnx"
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n✅  Done in {time.perf_counter() - t0:.1f}s")
    print(f"   File:  {out_path.name}  ({size_mb:.2f} MB)")
    if fp32_path.exists():
        fp32_mb = fp32_path.stat().st_size / (1024 * 1024)
        print(f"   FP32:  {fp32_path.name}  ({fp32_mb:.2f} MB)")
        print(f"   Ratio: {fp32_mb / size_mb:.1f}× size reduction\n")


if __name__ == "__main__":
    main()
