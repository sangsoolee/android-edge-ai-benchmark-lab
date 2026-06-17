# INT8 Quantization Accuracy — MobileNetV3-Small

**A measured negative finding.** Naive full-integer INT8 post-training quantization (PTQ)
reduced model size 3.5× but collapsed accuracy on ImageNet validation. This is kept as
an official benchmark result, not hidden — the point of this project is to measure when
optimization *fails*, not only when it succeeds.

- **Model:** MobileNetV3-Small (torchvision `IMAGENET1K_V1`)
- **Dataset:** ImageNet validation
- **Samples:** n = 500
- **Runtime:** LiteRT / TFLite via `ai-edge-litert` (CPU)
- **Metric:** Top-1 / Top-5

## Results

| Runtime | Precision | Quantization | Top-1 | Top-5 | Top-1 Drop | Result |
|---|---|---|---:|---:|---:|---|
| LiteRT / TFLite | FP32 | none | **69.4%** | 88.6% | — | ✅ Valid baseline |
| LiteRT / TFLite | FP16 | weight fp16 | **69.6%** | 88.6% | **+0.2 pp** | ✅ Safe compression |
| LiteRT / TFLite | INT8 | full-integer PTQ | **0.6%** | 1.6% | **−68.8 pp** | ❌ Collapsed |

> Top-5 collapsed too (88.6% → 1.6%), so this is not a near-miss ranking shift — the
> quantized model effectively stopped discriminating. FP16, by contrast, preserves
> accuracy (the +0.2 pp is within subset noise) at ~half the FP32 size — the safe
> middle option (no retraining).

## This is not an evaluation bug

A reader seeing 0.6% should first suspect the harness. These checks rule that out:

- **FP32 reached 69.4% Top-1** on the same pipeline (MobileNetV3-Small's official Top-1 is
  67.7%; 69.4% on a 500-image subset is in range) — preprocessing and label mapping are correct.
- **Identical manifest and preprocessing** were used for FP32 and INT8 (resize 256 → center-crop
  224 → torchvision normalization).
- **Label mapping was validated through the FP32 accuracy** (wnid → 0-based index from the
  sorted-wnid ordering that matches torchvision).
- **INT8 input dtype and quantization parameters (scale/zero-point) were read from the TFLite
  tensor metadata**; a head-to-head check confirmed FP32 predicts correctly while the INT8
  model's logits are compressed/near-constant on the same images.

## Likely causes (candidates, not confirmed)

- MobileNetV3 uses **quantization-sensitive components** — hard-swish / hard-sigmoid and
  squeeze-and-excitation blocks — whose dynamic range is hard for per-tensor full-integer PTQ.
- **Full-integer PTQ is sensitive** to representative-dataset quality and operator-level
  quantization behavior; the collapse persisted even with 500 real ImageNet calibration images.
- **Size reduction does not guarantee** latency improvement or accuracy preservation.

## Latency × memory × size × accuracy — the actual decision

Accuracy alone is not the decision; combined with the latency/memory results:

| Precision | Size | p50 latency | Memory | Top-1 | Decision |
|---|---:|---:|---:|---:|---|
| FP32 | 9.73 MB | 1.53 ms | ~100 MB | 69.4% | ✅ Selected |
| FP16 | 4.89 MB | — | ~100 MB | 69.6% | ✅ Safe compression |
| INT8 (full-integer) | 2.76 MB | 2.86 ms | ~100 MB | 0.6% | ❌ Rejected |

INT8 was 3.5× smaller on disk but **slower** (dequantize overhead on this chip) **and**
accuracy-collapsed — rejected despite the size win. **FP16 halves the size with no accuracy
loss and no retraining** — the practical middle option.

## v0.4.1b — QAT feasibility study

Can Quantization-Aware Training recover the collapsed INT8? Two findings:

1. **TFMOT does not support MobileNetV3.** `tfmot.quantization.keras.quantize_model` fails on
   Keras MobileNetV3Small (custom hard-swish / SE ops). So MobileNetV3 is hostile to
   quantization at *both* levels — PTQ collapse **and** no QAT-tooling support.
2. **QAT path verified on a quantization-friendly arch (Keras MobileNetV2), but PoC
   under-recovered.** FP32 sanity 72.6% Top-1 (path/labels OK). A 1-epoch / 500-image QAT
   fine-tune → full-integer INT8 reached only **16.6% Top-1** (non-collapsed, but far from
   recovery).

| Path | Precision | Top-1 | Top-5 | Status |
|---|---|---:|---:|---|
| Keras MobileNetV2 | FP32 | 72.6% | 90.6% | Sanity baseline |
| Keras MobileNetV2 | INT8 QAT (PoC) | 16.6% | 45.2% | Toolchain verified, under-recovered |

> **Limitation (disclosed):** the QAT PoC used the **same 500 images for fine-tune and eval**
> (no train/eval split), so 16.6% is an optimistic upper bound — and it was still low,
> confirming the PoC training budget was insufficient. This is a **toolchain-feasibility check,
> not an apples-to-apples recovery** of the MobileNetV3 baseline (different architecture/weights).

## Conclusion

For this MobileNetV3-Small classification case, **FP16 was the practical recovery path**: it
preserves accuracy while halving model size, with no retraining. Full-integer INT8 PTQ collapsed
accuracy and did not improve latency. A small QAT PoC (Keras MobileNetV2) verified the QAT
toolchain runs, but did not recover accuracy under the limited 500-image / 1-epoch setup —
proper QAT needs a real train/eval split and a larger training budget (**future work**).
This reinforces the project principle: **measure latency, memory, size, *and* accuracy together;
quantization success is model- and method-dependent, not a single switch.**

*Reproduce: `scripts/eval/accuracy_eval.py` (per model) → `scripts/eval/compare_accuracy.py` (diff → `summary.csv` + `summary.md`).*
