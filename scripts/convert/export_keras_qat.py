#!/usr/bin/env python3
"""
export_keras_qat.py
-------------------
v0.4.1b — QAT recovery proof-of-concept (MobileNetV2).

Context / why MobileNetV2 (not MobileNetV3):
  - v0.4 finding: torchvision MobileNetV3-Small → full-integer INT8 PTQ collapsed
    (69.4% → 0.6%).
  - We tried QAT on Keras MobileNetV3Small, but TF Model Optimization (TFMOT)
    `quantize_model` does NOT support it (custom hard-swish / SE ops). So
    MobileNetV3 is hostile to quantization at BOTH levels — PTQ collapse AND no
    QAT tooling support. That's itself a result.
  - To still demonstrate that a QAT *recovery path* exists, we use MobileNetV2
    (ReLU6, TFMOT-supported — the canonical QAT example). DIFFERENT architecture
    and weights from the baseline; report as a recovery-path demo, not as the
    MobileNetV3 baseline "fixed".

PoC: short fine-tune on a small ImageNet subset (the val manifest), 1–few epochs.
Not a full retrain. Goal: show INT8 accuracy is recoverable with QAT, vs PTQ collapse.

Requirements: pip install tensorflow tensorflow-model-optimization

Usage:
  python scripts/convert/export_keras_qat.py --manifest data/imagenet/val_manifest.csv --epochs 1 --limit 500

Output:
  models/mobilenet_v2_keras_fp32.tflite   (sanity baseline; eval --preprocess mobilenet_v2)
  models/mobilenet_v2_qat_int8.tflite      (QAT result;     eval --preprocess mobilenet_v2)
"""

import argparse
import csv
from pathlib import Path

import tensorflow as tf

OUTPUT_DIR = Path(__file__).parent.parent.parent / "models"
CROP = 224
RESIZE = 256


def _preprocess(path, label):
    """resize shorter→256, center-crop 224, then MobileNetV2 range [-1,1]."""
    img = tf.cast(tf.io.decode_jpeg(tf.io.read_file(path), channels=3), tf.float32)
    shape = tf.shape(img)
    h, w = shape[0], shape[1]
    scale = RESIZE / tf.cast(tf.minimum(h, w), tf.float32)
    nh = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
    nw = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)
    img = tf.image.resize(img, [nh, nw], method="bilinear")
    img = tf.image.resize_with_crop_or_pad(img, CROP, CROP)
    img = img / 127.5 - 1.0     # [0,255] -> [-1,1]
    return img, label


def make_dataset(manifest: Path, limit: int, batch: int) -> tf.data.Dataset:
    paths, labels = [], []
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            paths.append(row["image_path"]); labels.append(int(row["label_index"]))
            if limit and len(paths) >= limit:
                break
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def to_tflite_fp32(model, out: Path) -> None:
    out.write_bytes(tf.lite.TFLiteConverter.from_keras_model(model).convert())


def to_tflite_int8(model, rep_ds: tf.data.Dataset, out: Path) -> None:
    def rep():
        for imgs, _ in rep_ds.take(100):
            yield [imgs[:1]]
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.uint8
    c.inference_output_type = tf.uint8
    out.write_bytes(c.convert())


def main() -> None:
    p = argparse.ArgumentParser(description="QAT recovery PoC (Keras MobileNetV2 + TFMOT)")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    args = p.parse_args()

    try:
        import tensorflow_model_optimization as tfmot
    except ImportError:
        raise SystemExit("pip install tensorflow-model-optimization")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fixed input_shape so the exported TFLite has [1,224,224,3] (not dynamic).
    base = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=True, input_shape=(224, 224, 3)
    )

    print("\n[0/4] FP32 sanity export (validate preprocess + labels; expect ~71% Top-1)…")
    to_tflite_fp32(base, OUTPUT_DIR / "mobilenet_v2_keras_fp32.tflite")

    print("[1/4] TFMOT quantize_model (QAT wrap)…")
    q_model = tfmot.quantization.keras.quantize_model(base)
    q_model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                    loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    print(f"[2/4] Loading data (subset={args.limit})…")
    ds = make_dataset(args.manifest, args.limit, args.batch)

    print(f"[3/4] QAT fine-tune ({args.epochs} epoch, PoC)…")
    q_model.fit(ds, epochs=args.epochs)

    out = OUTPUT_DIR / "mobilenet_v2_qat_int8.tflite"
    print("[4/4] Exporting full-integer INT8…")
    to_tflite_int8(q_model, ds, out)

    print(f"\n✅  {out.name}  ({out.stat().st_size / (1024*1024):.2f} MB)")
    print("   Eval both with:  --preprocess mobilenet_v2\n")


if __name__ == "__main__":
    main()
