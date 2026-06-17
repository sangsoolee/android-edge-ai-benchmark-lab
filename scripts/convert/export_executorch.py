#!/usr/bin/env python3
"""
export_executorch.py
--------------------
Exports a PyTorch vision model to ExecuTorch .pte format for use with
ExecuTorch Android runtime.

Install:
  pip install executorch
  (ExecuTorch requires a matching torch version — check https://pytorch.org/executorch)

Usage:
  python export_executorch.py --model mobilenet_v3_small --precision fp32

Output:
  ../../models/<model>_<precision>.pte
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
    "efficientnet_b0":    lambda: tv_models.efficientnet_b0(weights="IMAGENET1K_V1"),
}

INPUT_SHAPE = (1, 3, 224, 224)
OUTPUT_DIR  = Path(__file__).parent.parent.parent / "models"

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_fp32(model: torch.nn.Module, model_name: str, backend: str) -> Path:
    """Export to .pte.

    backend="xnnpack": lower to the XNNPACK delegate — the production CPU path,
      typically an order of magnitude faster than the portable reference kernels.
    backend="portable": no delegate; portable reference implementation (slow, but
      dependency-free — useful as a baseline).
    """
    try:
        from torch.export import export
    except ImportError:
        raise SystemExit(
            "executorch not found.\n"
            "Install: pip install executorch\n"
            "See: https://pytorch.org/executorch/stable/getting-started-setup.html"
        )

    out = OUTPUT_DIR / f"{model_name}_fp32.pte"
    sample_inputs = (torch.randn(*INPUT_SHAPE),)

    print("  [1/3] torch.export.export ...")
    exported = export(model, sample_inputs)

    if backend == "xnnpack":
        try:
            from executorch.exir import to_edge_transform_and_lower
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
                XnnpackPartitioner,
            )
        except ImportError:
            raise SystemExit(
                "XNNPACK backend not available in this executorch install.\n"
                "Use --backend portable, or install an executorch build with XNNPACK."
            )
        print("  [2/3] to_edge_transform_and_lower (XNNPACK partitioner) ...")
        edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    else:
        from executorch.exir import to_edge
        print("  [2/3] to_edge (portable, no delegate) ...")
        edge = to_edge(exported)

    print("  [3/3] to_executorch ...")
    et_program = edge.to_executorch()

    with open(out, "wb") as f:
        f.write(et_program.buffer)

    print(f"  [ExecuTorch FP32 / {backend}] → {out}")
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(path: Path) -> None:
    """Run a single inference with the exported .pte via executorch runtime."""
    try:
        from executorch.runtime import Runtime as EtRuntime, Program
        import numpy as np

        runtime = EtRuntime.get()
        with open(path, "rb") as f:
            program = Program(runtime, f.read())

        method = program.load_method("forward")
        dummy = torch.randn(*INPUT_SHAPE)
        output = method.execute([dummy])
        print(f"  ✅ Validation OK — output shape: {output[0].shape}")
    except Exception as e:
        print(f"  ⚠️  Validation failed (runtime check optional): {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch model → ExecuTorch .pte")
    parser.add_argument("--model",     required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--precision", required=True, choices=["fp32"])
    parser.add_argument("--backend",   choices=["xnnpack", "portable"], default="xnnpack",
                        help="xnnpack = production CPU delegate (default); "
                             "portable = reference kernels (slow baseline)")
    parser.add_argument("--validate",  action="store_true", help="Run CPU inference check after export")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model:     {args.model}")
    print(f"  Precision: {args.precision}")
    print(f"  Backend:   {args.backend}")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()

    print("[1/2] Loading PyTorch model...")
    model = SUPPORTED_MODELS[args.model]()
    model.eval()

    print("[2/2] Exporting to ExecuTorch .pte ...")
    out_path = export_fp32(model, args.model, args.backend)

    if args.validate:
        print("\nValidating...")
        validate(out_path)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n✅  Done in {time.perf_counter() - t0:.1f}s")
    print(f"   File:  {out_path.name}")
    print(f"   Size:  {size_mb:.2f} MB\n")


if __name__ == "__main__":
    main()
