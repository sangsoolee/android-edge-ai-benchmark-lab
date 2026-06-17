#!/usr/bin/env python3
"""
cross_runtime_check.py
----------------------
Validates that the three conversion paths produce the *same* model, by running
each runtime on one identical fixed-seed input and comparing the 1000-class
logits pairwise.

Why: TFLite (PyTorch→ONNX→TF→TFLite), ONNX (PyTorch→ONNX), and ExecuTorch
(PyTorch→Edge→PTE) take different paths with layout rewrites and op fusion. Equal
latency means nothing if they don't compute the same thing — so we check
top-1 agreement, top-5 overlap, cosine similarity, and max-abs-error.

Layouts: ONNX/ExecuTorch take NCHW (1,3,224,224); TFLite takes NHWC (1,224,224,3).
The same canonical tensor is fed to each in its expected layout.

Requirements: numpy, ai-edge-litert, onnxruntime  (executorch optional)

Usage:
  python scripts/eval/cross_runtime_check.py \\
    --tflite models/mobilenet_v3_small_fp32.tflite \\
    --onnx   models/mobilenet_v3_small_fp32.onnx \\
    --pte    models/mobilenet_v3_small_fp32.pte      # optional

Exit code: non-zero if any pair's cosine similarity < --min-cosine (CI gate).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

SEED        = 42
INPUT_NCHW  = (1, 3, 224, 224)


def make_inputs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    x_nchw = rng.standard_normal(INPUT_NCHW).astype(np.float32)
    x_nhwc = np.transpose(x_nchw, (0, 2, 3, 1)).copy()  # (1,224,224,3)
    return x_nchw, x_nhwc


# ---------------------------------------------------------------------------
# Per-runtime inference (FP32 only — this is a correctness, not accuracy, check)
# ---------------------------------------------------------------------------

def run_tflite(path: Path, x_nhwc: np.ndarray) -> np.ndarray:
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    if inp["dtype"] != np.float32:
        raise SystemExit(f"{path.name}: expected FP32 input, got {inp['dtype']}")
    interp.set_tensor(inp["index"], x_nhwc)
    interp.invoke()
    return interp.get_tensor(out["index"]).reshape(-1)


def run_onnx(path: Path, x_nchw: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return np.array(sess.run(None, {name: x_nchw})[0]).reshape(-1)


def run_executorch(path: Path, x_nchw: np.ndarray) -> np.ndarray:
    import torch
    from executorch.runtime import Runtime as EtRuntime

    runtime = EtRuntime.get()
    program = runtime.load_program(str(path))
    method = program.load_method("forward")
    out = method.execute([torch.from_numpy(x_nchw)])
    return np.array(out[0]).reshape(-1)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(a: np.ndarray, b: np.ndarray) -> dict:
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    max_abs = float(np.max(np.abs(a - b)))
    top1_a, top1_b = int(a.argmax()), int(b.argmax())
    top5_a = set(np.argsort(a)[::-1][:5].tolist())
    top5_b = set(np.argsort(b)[::-1][:5].tolist())
    return {
        "cosine":       round(cos, 6),
        "max_abs_err":  round(max_abs, 6),
        "top1_match":   top1_a == top1_b,
        "top5_overlap": len(top5_a & top5_b),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-runtime output correctness check")
    parser.add_argument("--tflite", type=Path, default=None)
    parser.add_argument("--onnx",   type=Path, default=None)
    parser.add_argument("--pte",    type=Path, default=None)
    parser.add_argument("--min-cosine", type=float, default=0.999,
                        help="Fail (exit 1) if any pair's cosine < this (default 0.999)")
    args = parser.parse_args()

    x_nchw, x_nhwc = make_inputs()

    outputs: dict[str, np.ndarray] = {}
    runners = [
        ("tflite", args.tflite, lambda p: run_tflite(p, x_nhwc)),
        ("onnx",   args.onnx,   lambda p: run_onnx(p, x_nchw)),
        ("executorch", args.pte, lambda p: run_executorch(p, x_nchw)),
    ]
    for name, path, fn in runners:
        if path is None:
            continue
        if not path.exists():
            print(f"  skip {name}: {path} not found")
            continue
        try:
            outputs[name] = fn(path)
            print(f"  ran {name:11s} → logits[{outputs[name].shape[0]}]  "
                  f"top1={int(outputs[name].argmax())}")
        except Exception as e:
            print(f"  FAIL {name}: {e}")

    if len(outputs) < 2:
        raise SystemExit("Need at least two runtimes to compare.")

    print(f"\n{'pair':28s} {'cosine':>9} {'max_abs':>10} {'top1':>6} {'top5':>5}")
    print("  " + "-" * 58)
    names = list(outputs)
    worst_cos = 1.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = compare(outputs[names[i]], outputs[names[j]])
            worst_cos = min(worst_cos, r["cosine"])
            print(f"  {names[i]+' vs '+names[j]:26s} {r['cosine']:>9.5f} "
                  f"{r['max_abs_err']:>10.5f} {str(r['top1_match']):>6} {r['top5_overlap']:>4}/5")

    print()
    if worst_cos < args.min_cosine:
        print(f"❌ worst cosine {worst_cos:.5f} < {args.min_cosine} — runtimes disagree.")
        sys.exit(1)
    print(f"✅ all pairs agree (worst cosine {worst_cos:.5f} ≥ {args.min_cosine}).")


if __name__ == "__main__":
    main()
