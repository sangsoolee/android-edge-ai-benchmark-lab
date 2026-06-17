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
| LiteRT / TFLite | INT8 | full-integer PTQ | **0.6%** | 1.6% | **−68.8 pp** | ❌ Collapsed |

> Top-5 collapsed too (88.6% → 1.6%), so this is not a near-miss ranking shift — the
> quantized model effectively stopped discriminating.

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
| INT8 (full-integer) | 2.76 MB | 2.86 ms | ~100 MB | 0.6% | ❌ Rejected |

INT8 was 3.5× smaller on disk but **slower** (dequantize overhead on this chip) **and**
accuracy-collapsed — so it is rejected in this setup despite the size win.

## Conclusion

For this MobileNetV3-Small benchmark on the tested Android device, FP32 LiteRT with XNNPACK
provided the best practical trade-off. Full-integer INT8 PTQ significantly reduced model size,
but it did not improve latency and caused a catastrophic accuracy collapse. This reinforces the
core principle of the project: **edge AI optimization must be measured across latency, memory,
size, and accuracy together** — size or latency numbers alone can hide a broken model.

> Out of scope for v0.4 (tracked separately): QAT recovery and FP16 as an intermediate
> compression path — see the roadmap (v0.4.1).

*Reproduce: `scripts/eval/accuracy_eval.py` (per model) → `scripts/eval/compare_accuracy.py` (diff → `summary.csv` + `summary.md`).*
