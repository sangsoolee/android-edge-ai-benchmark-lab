#!/usr/bin/env python3
"""
accuracy_eval.py
----------------
Evaluates top-1 / top-5 ImageNet accuracy for a single TFLite model.
Run once for FP32 and once for INT8, then pass both JSONs to compare_accuracy.py.

Requirements:
  pip install ai-edge-litert pillow numpy tqdm

Usage:
  # Smoke test (first 500 images)
  python scripts/eval/accuracy_eval.py \\
    --model models/mobilenet_v3_small_fp32.tflite \\
    --manifest data/imagenet/val_manifest.csv \\
    --limit 500

  # Full 5k evaluation
  python scripts/eval/accuracy_eval.py \\
    --model models/mobilenet_v3_small_fp32.tflite \\
    --manifest data/imagenet/val_manifest.csv \\
    --limit 5000

  python scripts/eval/accuracy_eval.py \\
    --model models/mobilenet_v3_small_int8.tflite \\
    --manifest data/imagenet/val_manifest.csv \\
    --limit 5000

Output:
  results/accuracy/<model_stem>_results.json

val_manifest.csv columns (built by prepare_imagenet_val.py):
  image_path, label_index, wnid, class_name
  label_index is 0-based and matches model output (torchvision ordering).
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESIZE_SIZE   = 256
CROP_SIZE     = 224


def _find_imagenet_class_index_json() -> "Path | None":
    """Locate imagenet_class_index.json in the installed torchvision package."""
    import torchvision
    base = Path(torchvision.__file__).parent
    # importlib.resources (Python 3.9+) — works regardless of physical layout
    try:
        import importlib.resources as pkg
        with pkg.as_file(pkg.files("torchvision") / "data" / "imagenet_class_index.json") as p:
            if p.exists():
                return p
    except Exception:
        pass
    hits = list(base.rglob("imagenet_class_index.json"))
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(image_path: str, mode: str = "torchvision") -> np.ndarray:
    """Load and preprocess one image (resize 256 → center crop 224).

    mode="torchvision": ToTensor + mean/std normalize (matches torchvision
      IMAGENET1K_V1 weights — the FP32/FP16/PTQ-INT8 torchvision→TFLite models).
    mode="keras": values stay in RGB [0,255], NO normalization (matches
      tf.keras.applications.MobileNetV3Small with include_preprocessing=True,
      i.e. the QAT model — its Rescaling layer is baked in).

    Returns float32 NHWC array of shape (1, 224, 224, 3).
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")

    # Resize shorter side to 256
    w, h = img.size
    scale = RESIZE_SIZE / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop 224×224
    left = (new_w - CROP_SIZE) // 2
    top  = (new_h - CROP_SIZE) // 2
    img  = img.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))

    arr = np.array(img, dtype=np.float32)                    # (224, 224, 3) in [0,255]
    if mode == "keras":
        return arr[np.newaxis]                               # model rescales internally
    if mode == "mobilenet_v2":
        return (arr / 127.5 - 1.0)[np.newaxis]               # MobileNetV2 range [-1,1]
    arr = arr / 255.0                                        # [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD               # normalize
    return arr[np.newaxis]                                   # (1, 224, 224, 3) NHWC


# ---------------------------------------------------------------------------
# Input quantization
# ---------------------------------------------------------------------------

def quantize_input(float_nhwc: np.ndarray, input_detail: dict) -> np.ndarray:
    """Convert normalized float32 to the dtype the interpreter expects.

    For UINT8 / INT8 full-integer models:
      q = round(float_val / scale + zero_point), clamped to dtype range.
    For FP32 models: pass through unchanged.
    """
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        return float_nhwc

    quant_params = input_detail.get("quantization_parameters", {})
    scales      = quant_params.get("scales", [])
    zero_points = quant_params.get("zero_points", [])

    if len(scales) == 0 or scales[0] == 0:
        # Fallback: scale to [0,255] assuming input is roughly normalized
        if dtype == np.uint8:
            arr = (float_nhwc * 128 + 128).clip(0, 255).astype(np.uint8)
        else:
            arr = (float_nhwc * 128).clip(-128, 127).astype(np.int8)
        return arr

    scale      = float(scales[0])
    zero_point = int(zero_points[0])
    quantized  = float_nhwc / scale + zero_point

    if dtype == np.uint8:
        return np.clip(np.round(quantized), 0, 255).astype(np.uint8)
    else:  # int8
        return np.clip(np.round(quantized), -128, 127).astype(np.int8)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def detect_precision(model_path: Path, input_dtype: type) -> str:
    stem = model_path.stem.lower()
    if "int8" in stem:
        return "int8"
    if "fp16" in stem:
        return "fp16"
    if input_dtype == np.uint8 or input_dtype == np.int8:
        return "int8"
    return "fp32"


def load_manifest(manifest_path: Path, limit: Optional[int]) -> list[dict]:
    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "image_path":  row["image_path"],
                "label_index": int(row["label_index"]),
            })
            if limit and len(rows) >= limit:
                break
    return rows


def evaluate(
    model_path: Path,
    manifest: list[dict],
    output_path: Path,
    preprocess_mode: str = "torchvision",
) -> dict:
    try:
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
    except ImportError:
        raise SystemExit(
            "pip install ai-edge-litert\n"
            "(or: pip install tflite-runtime  — then change the import in this script)"
        )

    try:
        from tqdm import tqdm
        progress = lambda x, **kw: tqdm(x, **kw)
    except ImportError:
        progress = lambda x, **kw: x

    # XNNPACK (the default CPU delegate) cannot prepare some full-integer INT8
    # graphs and fails in allocate_tensors(). Fall back to the reference kernels
    # (BUILTIN_WITHOUT_DEFAULT_DELEGATES) when that happens.
    try:
        interp = Interpreter(model_path=str(model_path))
        interp.allocate_tensors()
    except RuntimeError as e:
        print(f"  ⚠️  XNNPACK delegate failed ({e.__class__.__name__}); "
              "retrying without default delegates…")
        interp = Interpreter(
            model_path=str(model_path),
            experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
        interp.allocate_tensors()

    input_details  = interp.get_input_details()
    output_details = interp.get_output_details()
    input_detail   = input_details[0]
    output_detail  = output_details[0]
    input_dtype    = input_detail["dtype"]
    precision      = detect_precision(model_path, input_dtype)

    print(f"\n[accuracy_eval] {model_path.name}")
    print(f"  Input  : {input_detail['shape']}  dtype={input_dtype.__name__}")
    print(f"  Output : {output_detail['shape']}  dtype={output_detail['dtype'].__name__}")
    print(f"  Samples: {len(manifest)}")

    top1_correct = top5_correct = 0
    errors = 0
    t0 = time.perf_counter()

    for item in progress(manifest, desc=model_path.stem, unit="img"):
        try:
            img_nhwc = preprocess(item["image_path"], preprocess_mode)
        except Exception as e:
            errors += 1
            continue

        model_input = quantize_input(img_nhwc, input_detail)
        interp.set_tensor(input_detail["index"], model_input)
        interp.invoke()

        raw_output = interp.get_tensor(output_detail["index"])  # (1, ..., 1000)
        logits     = raw_output.reshape(-1)                     # flatten to (1000,)
        top5_idx   = np.argsort(logits)[::-1][:5]

        true_label = item["label_index"]
        if top5_idx[0] == true_label:
            top1_correct += 1
        if true_label in top5_idx:
            top5_correct += 1

    elapsed   = time.perf_counter() - t0
    n_counted = len(manifest) - errors

    result = {
        "model":         model_path.name,
        "precision":     precision,
        "top1_accuracy": round(top1_correct / n_counted, 6) if n_counted else 0.0,
        "top5_accuracy": round(top5_correct / n_counted, 6) if n_counted else 0.0,
        "top1_correct":  top1_correct,
        "top5_correct":  top5_correct,
        "sample_count":  n_counted,
        "errors":        errors,
        "elapsed_sec":   round(elapsed, 2),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Top-1: {result['top1_accuracy'] * 100:.2f}%  "
          f"({top1_correct}/{n_counted})")
    print(f"  Top-5: {result['top5_accuracy'] * 100:.2f}%  "
          f"({top5_correct}/{n_counted})")
    print(f"  Time : {elapsed:.1f}s  ({elapsed / max(n_counted, 1) * 1000:.1f} ms/img)")
    print(f"\n  → {output_path}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).parents[2]

    parser = argparse.ArgumentParser(
        description="Evaluate TFLite model top-1/top-5 accuracy on ImageNet"
    )
    parser.add_argument("--model", required=True, type=Path,
                        help="Path to .tflite model file")
    parser.add_argument("--manifest", type=Path,
                        default=repo_root / "data" / "imagenet" / "val_manifest.csv",
                        help="val_manifest.csv from prepare_imagenet_val.py")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max images to evaluate (default: all ~50k)")
    parser.add_argument("--preprocess", choices=["torchvision", "keras", "mobilenet_v2"],
                        default="torchvision",
                        help="torchvision = mean/std normalize (torchvision models); "
                             "keras = RGB [0,255] (Keras include_preprocessing models); "
                             "mobilenet_v2 = [-1,1] (Keras MobileNetV2 QAT model)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: results/accuracy/<model>_results.json)")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    if not args.manifest.exists():
        raise SystemExit(
            f"Manifest not found: {args.manifest}\n"
            "Run: python scripts/eval/prepare_imagenet_val.py --help"
        )

    output_path = args.output or (
        repo_root / "results" / "accuracy" / f"{args.model.stem}_results.json"
    )

    manifest = load_manifest(args.manifest, args.limit)
    print(f"Loaded {len(manifest)} entries from manifest")

    evaluate(args.model, manifest, output_path, preprocess_mode=args.preprocess)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
