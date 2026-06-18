# What I learned measuring AI inference runtimes on a real Android phone

> A reproducible benchmark of LiteRT, ONNX Runtime, and ExecuTorch on a Galaxy S26
> Ultra (Snapdragon 8 Gen 3, Android 16) — and why almost every "on-device AI"
> rule of thumb broke once I measured it.

Most "on-device AI" write-ups quote a single average latency, run on an emulator,
or never disclose the measurement conditions. I wanted the opposite: a pipeline
where every number is reproducible, the methodology is enforced in code, and the
raw CSVs are committed so anyone can check my work.

This post is the story of what that produced. The short version: **"GPU is
faster", "INT8 is faster", "ExecuTorch is fast" are all false in general** — each
is true only under conditions you have to measure on your target device.

---

## How the measurement works

The app loads a model, runs it under a fixed protocol, and writes a CSV row:

- **warmup = 20** (discarded), **measured = 100** runs
- **p50 / p95 / p99** latency (`SystemClock.elapsedRealtimeNanos()`), cold start,
  peak PSS memory, thermal status before/after
- **release** build, **arm64-v8a**, fixed input tensor
- manual protocol the experimenter controls: airplane mode, 50% brightness, no
  charging, 5-min cooldown

I'm explicit about what's enforced vs manual — the app can't force you to pull the
charging cable, so I don't claim it does.

**Why p99?** p50 looks great on a slide; p99 is what a user feels on a bad thermal
day. A runtime with `p50=12ms` but `p99=180ms` is not production-ready.

**Correctness first.** Before trusting any latency gap, I verify the runtimes
compute the *same* model: feeding one fixed-seed input, the LiteRT / ONNX /
ExecuTorch outputs match to cosine ≈ 1.000. So the latency differences are pure
runtime/backend efficiency, not divergent conversions.

---

## Finding 1 — "GPU is faster" is not true for small models

MobileNetV3-Small, same model, three precisions/backends on LiteRT:

| Backend | Precision | p50 | Cold start | Peak mem |
|---|---|---:|---:|---:|
| **CPU (XNNPACK)** | FP32 | **1.53 ms** | 18 ms | 100 MB |
| GPU delegate | FP32 | 2.59 ms | **177 ms** | 187 MB |
| CPU (XNNPACK) | INT8 | 2.86 ms | 16 ms | 100 MB |

The GPU delegate was **slower** than CPU here, with a **~177 ms cold start** (≈10×
the CPU path) from shader compilation. For a small model that runs in ~1.5 ms on
CPU, the GPU's setup and dispatch overhead never pays off. GPU wins on big models;
it loses on small ones. You have to know which regime you're in.

---

## Finding 2 — the runtime gap is mostly a *configuration* gap

Out of the box, the three runtimes look very different on MobileNetV3-Small:

| Runtime | Backend | p50 |
|---|---|---:|
| LiteRT | CPU (XNNPACK) | 1.53 ms |
| ONNX Runtime | CPU (MLAS, default) | 5.43 ms |
| ExecuTorch | CPU (portable) | 62.6 ms |

LiteRT looks 3.5× faster than ONNX and ~40× faster than ExecuTorch. But that's not
a fair fight — it's XNNPACK vs ONNX's default MLAS path vs ExecuTorch's *portable
reference* kernels. So I enabled each runtime's accelerator:

- **ONNX NNAPI**: median dropped 5.43 → **3.28 ms** (~40% faster) on Snapdragon —
  but bimodal (partial CPU fallback for hard-swish), and on Exynos NNAPI was
  actually *slower* than its own CPU path (more below).
- **ExecuTorch XNNPACK delegate**: exporting a `.pte` with the XNNPACK partitioner
  takes ExecuTorch from **70.8 ms (portable) → 0.89 ms** — the fastest config in the
  whole study, narrowly ahead of LiteRT (1.48 ms). Both run XNNPACK kernels, so that
  parity makes sense.

> **A correction I have to own.** An earlier version of this write-up reported
> ExecuTorch as "44× slower / XNNPACK didn't help (74 ms)." That was wrong, in two
> compounding ways: I was measuring the **portable executor**, and on one device a
> **stale portable `.pte` was still on disk** while I thought I was testing the
> XNNPACK one. The model size in the CSV (9.84 MB portable vs 9.73 MB XNNPACK) is
> what gave it away. Once the right artifact was on-device, ExecuTorch was ~0.9 ms.
> **The lesson is the most useful one in the project: verify the artifact that is
> actually on the device — not the one you think you pushed.**

---

## Finding 3 — INT8 is not universally good (this surprised me most)

I measured INT8 accuracy on ImageNet validation, not just latency. The result was
a clean negative finding:

| MobileNetV3-Small | Top-1 | Size | Verdict |
|---|---:|---:|---|
| FP32 | 69.4% | 9.73 MB | baseline |
| **FP16** | **69.6%** | 4.89 MB | safe compression |
| INT8 (full-integer PTQ) | **0.6%** | 2.76 MB | **collapsed** |

Naive full-integer INT8 post-training quantization **destroyed accuracy** (69.4% →
0.6%, Top-5 fell too) while *also* being slower than FP32 on this CPU. The FP32
baseline hitting 69.4% (vs the official 67.7%) proves the harness is correct — this
is a real collapse, not an eval bug. Likely cause: MobileNetV3's hard-swish and
squeeze-excite blocks are hostile to per-tensor full-integer PTQ.

**FP16 was the actual answer**: accuracy preserved, half the size, no retraining.

Then I asked whether QAT could recover INT8 — and hit a second wall: TensorFlow's
QAT tooling (TFMOT) **doesn't support MobileNetV3** at all (custom ops). On a
quantization-friendly arch (MobileNetV2) the QAT toolchain runs, but my small
proof-of-concept (1 epoch, 500 images) only reached 16.6% — far from recovery. I'm
reporting that honestly as *toolchain feasibility, not recovery*; proper QAT needs
a real training budget. **MobileNetV3 turned out hostile to quantization at every
level — PTQ, and even the QAT tooling.**

### …but for YOLOv8n, INT8 was a big win

Same INT8, completely different verdict on a detection model:

| YOLOv8n (LiteRT CPU) | Inference p50 |
|---|---:|
| FP32 | 83.6 ms |
| INT8 | **26.8 ms** (3.1× faster) |

On the heavy 640×640 detector, INT8's compute savings dominate. So INT8 *hurt* the
tiny classifier and *helped* the heavy detector. **INT8's value is a function of
the compute-to-overhead ratio — there is no universal answer.**

---

## Finding 4 — measuring only "inference" lies to you

For YOLOv8n I split the on-device pipeline into **preprocess / inference /
postprocess**, each with its own timer (plus an independent end-to-end timer,
because the p50 of end-to-end ≠ the sum of phase p50s):

| Precision | Preprocess | Inference | Postprocess | End-to-End |
|---|---:|---:|---:|---:|
| FP32 | 4.1 ms | 83.6 ms | 1.0 ms | **88.7 ms** |
| INT8 | 10.1 ms | **26.8 ms** | 2.5 ms | **39.3 ms** |

On inference alone, INT8 is 3.1× faster. But INT8 pays for it elsewhere: float→int8
**input quantization makes preprocess ~2.5× slower**, and output dequant doubles
postprocess. End-to-end, INT8's lead shrinks to **2.25×**. An inference-only
benchmark would have overstated the real-world gain by ~40%. If you optimize the
number you measure, measure the number the user actually waits for.

---

## Finding 5 — "fastest runtime" is a per-device question

I ran the identical APK/models/protocol on a second phone — **Exynos 2400
(Xclipse GPU)** alongside the **Snapdragon 8 Gen 3 (Adreno)**. Median p50 (ms),
MobileNetV3-Small unless noted:

| Runtime / Backend | Prec | Snapdragon | Exynos |
|---|---|---:|---:|
| ExecuTorch CPU (XNNPACK) | FP32 | **0.89** | **1.13** |
| LiteRT CPU (XNNPACK) | FP32 | 1.48 | 1.67 |
| ONNX CPU | FP32 | 5.43 | **3.39** |
| ONNX NNAPI | FP32 | **3.28** | 4.95 |
| LiteRT CPU — YOLOv8n | FP32 | 87.9 | 142.9 |
| LiteRT CPU — YOLOv8n | INT8 | 27.0 | 38.5 |

What survived the chip swap and what didn't:

- **NNAPI flipped.** It *helped* ONNX on Snapdragon (5.43 → 3.28) and *hurt* it on
  Exynos (3.39 → 4.95, slower than that chip's own CPU). "Turn on the NPU" is not
  portable advice — it depends on the vendor's driver and which ops it accepts.
- **Chip strengths are workload-shaped.** Exynos was *faster* on the small ONNX-CPU
  path but *slower* on the heavy YOLOv8n. There's no single "faster phone."
- **The headline held:** ExecuTorch-XNNPACK fastest, and INT8 helping YOLO, were
  true on both. But the runner-up ordering reshuffled per device.

The pipeline groups results by `device_model`, so a third phone is a
measure-and-drop-in. Raw CSVs: [`results/raw/`](../../results/raw/).

---

## What I'd tell my past self

- **Measure on your target device.** Every universal claim ("GPU/INT8/runtime X is
  faster") broke under measurement.
- **Measure accuracy alongside latency/size.** INT8 made a model 3.5× smaller and
  0.6% accurate. Size dashboards would have called that a win.
- **Split the pipeline.** Inference-only numbers hide quantize/dequant cost.
- **Verify equivalence before comparing.** Cosine ≈ 1.000 across runtimes is what
  makes the latency comparison meaningful.
- **Report negative and inconclusive results honestly.** The INT8 collapse and the
  under-recovered QAT PoC are some of the most useful things in this repo.

Everything here — app, conversion scripts, raw CSVs — is reproducible from the
[repository](../../README.md).
