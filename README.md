# Android AI Benchmark — On-Device Inference Runtime Comparison

> **Does GPU always beat CPU? Does INT8 always beat FP32?**  
> We measured it on a real device. The answers might surprise you.

[![Android](https://img.shields.io/badge/Platform-Android%208.0+-green.svg)](https://developer.android.com)
[![Kotlin](https://img.shields.io/badge/Language-Kotlin-purple.svg)](https://kotlinlang.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Build](https://img.shields.io/github/actions/workflow/status/sangsoolee/android-edge-ai-benchmark-lab/build.yml?label=APK%20build)](https://github.com/sangsoolee/android-edge-ai-benchmark-lab/actions)

---

<p align="center">
  <img src="docs/screenshot_v02.png" width="320" alt="Benchmark results on Galaxy S26 Ultra"/>
</p>

---

## What We Found (Galaxy S26 Ultra · Snapdragon 8 Gen 3)

### MobileNetV3-Small — All Runtimes

| Runtime | Backend | Precision | Model Size | p50 | p95 | p99 | Cold Start | Memory |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **LiteRT** | **CPU (XNNPACK)** | **FP32** | 9.73 MB | **1.53 ms** ✅ | 1.55 ms | 1.56 ms | 18 ms | 100 MB |
| LiteRT | GPU delegate | FP32 | 9.73 MB | 2.59 ms | 3.23 ms | 3.70 ms | 177 ms | 187 MB |
| LiteRT | CPU (XNNPACK) | INT8 | 2.76 MB | 2.86 ms | 3.22 ms | 3.93 ms | 16 ms | 100 MB |
| ONNX Runtime | NNAPI | FP32 | 9.71 MB | 3.28 ms | 5.58 ms | 5.73 ms | 39 ms | 143 MB |
| ONNX Runtime | CPU | FP32 | 9.71 MB | 5.43 ms | 5.68 ms | 5.85 ms | 44 ms | 147 MB |
| ExecuTorch | CPU (portable) | FP32 | 9.84 MB | 62.6 ms | 63.3 ms | 63.5 ms | 99 ms | 114 MB |
| ExecuTorch | CPU (XNNPACK-lowered) | FP32 | 9.73 MB | 74.1 ms ⚠️ | 75.7 ms | 75.8 ms | 89 ms | 140 MB |

> Measurement: warmup=20, measured=100 runs · **median of 5 sessions** · release build · airplane mode · no charging · 5-min cooldown · fixed-seed synthetic input (outputs verified identical across runtimes — see below)

### MobileNetV3-Small — FP32 vs INT8 Accuracy (ImageNet validation, n=500)

| Runtime | Precision | Quantization | Top-1 | Top-5 | Top-1 Drop | Result |
|---|---|---|---:|---:|---:|---|
| LiteRT / TFLite | FP32 | none | **69.4%** | 88.6% | — | ✅ Valid baseline |
| LiteRT / TFLite | INT8 | full-integer PTQ | **0.6%** | 1.6% | **−68.8 pp** | ❌ Accuracy collapse |

**This is a measured negative finding, not an eval bug.** FP32 reaching 69.4% (vs MobileNetV3-Small's official 67.7%) confirms preprocessing and label mapping are correct; the same manifest, preprocessing, and dtype detection were used for both precisions. *In this setup*, naive full-integer INT8 PTQ collapsed accuracy (Top-5 fell too: 88.6% → 1.6%) while only shrinking the model. See [results/accuracy/README.md](results/accuracy/README.md) for the sanity checks and likely causes.

**Latency × size × accuracy — the actual decision:**

| Precision | Size | p50 latency | Memory | Top-1 | Decision |
|---|---:|---:|---:|---:|---|
| FP32 | 9.73 MB | 1.53 ms | ~100 MB | 69.4% | ✅ Selected |
| INT8 (full-integer) | 2.76 MB | 2.86 ms | ~100 MB | 0.6% | ❌ Rejected |

> INT8 was 3.5× smaller but **slower** (dequantize overhead on this chip) **and** accuracy-collapsed — rejected here despite the size win. **Edge optimization must be measured across latency, memory, size, *and* accuracy together** — size/latency alone can hide a broken model.

**Key findings:**
- 🏆 **LiteRT CPU FP32 is fastest (1.53 ms)** — Snapdragon 8 Gen 3 + XNNPACK. All three runtimes produce identical outputs (cosine ≈ 1.000, same top-1), so the gaps are pure runtime/backend efficiency — not divergent conversions.
- ⚡ **ONNX NNAPI beats ONNX CPU by ~40% at the median (5.43 → 3.28 ms)** — enabling the on-device accelerator helps, but the distribution is bimodal: NNAPI partially falls back to CPU for MobileNetV3's hard-swish ops, so p95/p99 stay near the CPU path.
- 🐢 **ExecuTorch XNNPACK lowering did NOT help here (62.6 → 74.1 ms)** — on the prebuilt `executorch-android:1.3.1` AAR we could not confirm the XNNPACK delegate engaged (no delegate logs, no speedup, measured under back-to-back thermal load). Lowering the `.pte` is necessary but not sufficient — production use needs a runtime build with the XNNPACK backend verified.
- ⚠️ **GPU cold start ≈ 177 ms** — ~10× the CPU path (shader compilation). Critical for first-launch UX.
- ⚠️ **INT8 is ~2× slower than FP32 on CPU (1.53 → 2.86 ms)** — dequantize ops outweigh compute savings on this chip; INT8's only win here is 3.5× smaller on disk.

*The "fair tier" lesson: enabling each runtime's accelerator (NNAPI, XNNPACK) did **not** universally close the gap — it helped ONNX modestly and didn't help ExecuTorch at all on this build. "GPU is faster", "INT8 is faster", "this accelerator is faster" are not universal truths — always benchmark on your target device.*

### YOLOv8n — Object Detection (inference-only latency)

640×640 input, same protocol (median of 5). **Inference only** — preprocess (letterbox) and postprocess (NMS) are split out separately in v0.5.2–0.5.3.

| Runtime | Backend | Precision | Size | p50 | p95 | p99 | Cold Start | Memory |
|---|---|---|---:|---:|---:|---:|---:|---:|
| LiteRT | CPU (XNNPACK) | FP32 | 12.28 MB | 87.9 ms | 89.0 ms | 89.5 ms | 116 ms | 159 MB |
| LiteRT | CPU (XNNPACK) | INT8 | 3.27 MB | **27.0 ms** | 27.5 ms | 27.6 ms | 51 ms | 146 MB |

**On YOLOv8n, INT8 is 3.3× faster *and* 3.75× smaller — the opposite of MobileNetV3-Small, where INT8 was *slower*.** On the heavy 640×640 detector the INT8 compute savings dominate; on the tiny classifier the dequantize overhead outweighed them. INT8's benefit is a function of the **compute-to-overhead ratio**, not a universal rule.

> ⚠️ Latency only — **YOLOv8n INT8 detection accuracy (mAP) is not yet verified.** After the MobileNetV3 INT8 collapse, a *fast* INT8 model is not automatically a *correct* one; mAP validation is future work.

### YOLOv8n — Detection postprocess (Python ↔ Android parity)

The full detection pipeline (letterbox → inference → decode → class-aware NMS) runs on-device and is verified against the Python reference on the same image:

<p align="center">
  <img src="docs/yolov8n_detection_sample.png" width="360" alt="YOLOv8n detections on the bus sample (on-device)"/>
</p>

| | Python (reference) | Android (on-device) |
|---|---|---|
| detections | 4 person + 1 bus | 4 person + 1 bus |
| score Δ | — | ≤ 0.006 |
| box Δ | — | ≤ ~4 px (bus ~10 px, 1.3%) |

Identical class IDs, count, and order; the sub-percent box differences trace to the bilinear-resize implementation (Android `createScaledBitmap` vs PIL), not the postprocess logic. Reproduce: [`scripts/eval/yolo_detect_reference.py`](scripts/eval/yolo_detect_reference.py) (Python) and the app's **Detect Sample** button (Android).

### YOLOv8n — 3-phase latency breakdown (FP32 vs INT8)

Same image, 640×640, LiteRT CPU, median of 100 (warmup 20). Each phase is timed separately; **end-to-end has its own outer timer** (so its p50 ≠ the sum of phase p50s). Preprocess includes the INT8 input quantization; postprocess includes the INT8 output dequant. Rendering is excluded.

| Precision | Preprocess | Inference | Postprocess | End-to-End | Size | Peak Mem |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 4.1 ms | 83.6 ms | 1.0 ms | **88.7 ms** | 12.28 MB | 198 MB |
| INT8 | 10.1 ms | **26.8 ms** | 2.5 ms | **39.3 ms** | 3.27 MB | 146 MB |

**The phase split changes the conclusion.** On inference alone INT8 is **3.1× faster** (83.6 → 26.8 ms). But it pays for that elsewhere: float→int8 **input quantization makes preprocess ~2.5× slower** (4.1 → 10.1 ms) and output dequant doubles postprocess. **End-to-end, INT8's lead shrinks to 2.25×** (88.7 → 39.3 ms) — still a clear win here, but an *inference-only* benchmark would have overstated it by ~40%.

> The preprocess penalty is partly a scalar float→int8 loop (optimizable with a vectorized/NEON path); the postprocess penalty is the output dequant. Either way the lesson holds: **quantization relocates cost into pre/post — measure every phase, not just inference.**

**v0.4 ↔ v0.5, the through-line:**
- MobileNetV3-Small: INT8 PTQ **collapsed accuracy and was slower** → rejected.
- YOLOv8n: INT8 is **genuinely faster end-to-end (2.25×)** — but less than inference-only implies.
- INT8's value is entirely **model-, runtime-, and phase-dependent.** There is no universal answer; you have to measure.

---

## What This Project Is

A **reproducible benchmarking pipeline** for comparing on-device AI inference runtimes on real Android hardware.

Most "on-device AI" articles benchmark on a simulator, use a single average latency number, or don't disclose measurement conditions. This project measures **p50/p95/p99 latency, cold-start time, peak PSS memory, and thermal status**, documents the exact measurement protocol, and commits the raw result rows — every benchmark run, including repeat sessions — to [results/raw/](results/raw/). (Each row aggregates one 100-inference run; committing the full per-inference distributions is planned — see `docs`/roadmap.)

**Runtimes compared:**

| Runtime | Status | Backends |
|---|---|---|
| [LiteRT / TFLite](https://ai.google.dev/edge/litert) 2.x | ✅ v0.1 | CPU (XNNPACK), GPU delegate |
| [ONNX Runtime Android](https://onnxruntime.ai) 1.x | ✅ v0.2 | CPU, NNAPI |
| [ExecuTorch](https://pytorch.org/executorch) 1.x | ✅ v0.3 | CPU (portable / XNNPACK-lowered) |

---

## Reproducibility First

Measurement conditions are documented and **partially recorded in code**. The app automatically controls and captures the warmup/measured loops, cold-start, thermal status, and peak memory. The remaining conditions (airplane mode, brightness, charging, cooldown) are a **manual protocol the experimenter must follow** — they are not enforced by the app.

```
Recorded / enforced in code:
  Warmup runs:    20 (discarded)
  Measured runs:  100
  Input:          Fixed input tensor, identical every run
  Cold start:     model load + first inference
  Thermal:        PowerManager status before/after
  Peak memory:    PSS polling during measured loop

Manual protocol (experimenter-controlled, not enforced):
  Build type:     Release (never debug — interpreter overhead is significant)
  ABI:            arm64-v8a only
  Network:        Airplane mode ON
  Screen:         50% brightness, fixed
  Charging:       Cable disconnected
  Cooldown:       5 min idle before each session
```

### Fair-tier comparison & correctness

Beyond the default CPU path, each runtime is measured with its on-device accelerator where available (LiteRT GPU delegate, ONNX NNAPI, ExecuTorch XNNPACK lowering). Before comparing latency, the three runtimes are verified to compute the **same** model: one fixed-seed input yields matching logits (cosine ≈ 1.000, identical top-1) — see [`scripts/eval/cross_runtime_check.py`](scripts/eval/cross_runtime_check.py). This ensures latency gaps reflect runtime efficiency, not divergent conversions.

### Why p99?

p50 looks great on a chart. p99 is what your users actually experience on a bad thermal day.  
A runtime with `p50=12ms` but `p99=180ms` is not production-ready.

> **Percentile definition:** computed as floor-index nearest-rank — `sorted[floor(p/100 · (n-1))]`. With `n=100`, p99 maps to index 98 (the 99th of 100 sorted samples), i.e. effectively the near-maximum rather than an interpolated value.

---

## How to Reproduce

### Step 1 — Python environment

```bash
cd android-edge-ai-benchmark-lab

python3 -m venv .venv-convert
source .venv-convert/bin/activate       # Windows: .venv-convert\Scripts\activate

pip install --upgrade pip setuptools wheel
pip install cmake ninja
pip install -r scripts/convert/requirements.txt
pip install tf-keras
```

> The conversion stack (`torch` + `tensorflow` + `onnx2tf`) has conflicting version constraints. A dedicated venv is required.

### Step 2 — Convert models

```bash
# TFLite (LiteRT)
python scripts/convert/export_tflite.py --model mobilenet_v3_small --precision fp32
# INT8 with random calibration (fast)
python scripts/convert/export_tflite.py --model mobilenet_v3_small --precision int8
# INT8 with real ImageNet calibration (better accuracy, requires dataset)
python scripts/convert/export_tflite.py --model mobilenet_v3_small --precision int8 \
  --representative-data /path/to/imagenet/val --representative-samples 500

# ONNX
python scripts/convert/export_onnx.py --model mobilenet_v3_small --precision fp32

# ExecuTorch
python scripts/convert/export_executorch.py --model mobilenet_v3_small --precision fp32

# TorchAO INT8 (alternative quantization pipeline)
python scripts/convert/export_torchao.py --model mobilenet_v3_small
```

### Step 3 — Push to device

```bash
adb shell mkdir -p /sdcard/Android/data/com.edgeai.benchmark/files/models
adb push models/ /sdcard/Android/data/com.edgeai.benchmark/files/models/
```

### Step 4 — Build & install

```bash
./gradlew assembleRelease
adb install -r app/build/outputs/apk/release/app-release.apk
```

### Step 5 — Run & pull results

Run the app → select runtime/model/backend/precision → **Run Benchmark**

```bash
# Pull results. Note: adb nests them under results/raw/results/ — move the CSV
# up to results/raw/ (the committed location the analyze scripts expect).
adb pull /sdcard/Android/data/com.edgeai.benchmark/files/results/ ./results/raw/
mv ./results/raw/results/*.csv ./results/raw/ 2>/dev/null || true
```

### Step 6 — Analyze

```bash
# Charts (latency / memory / cold-start)
python scripts/analyze/plot_results.py \
  --input results/raw/ \
  --output results/graphs/

# Per-configuration summary table
python scripts/analyze/parse_results.py \
  --input results/raw/ \
  --markdown

# FP32 vs INT8 accuracy (requires ImageNet validation set)
# Step A: build manifest from Kaggle download
python scripts/eval/prepare_imagenet_val.py \
  --val-images-dir /path/to/ILSVRC/Data/CLS-LOC/val \
  --solution-csv   /path/to/LOC_val_solution.csv \
  --output-dir     data/imagenet

# Step B: evaluate each model (smoke test: --limit 500)
python scripts/eval/accuracy_eval.py \
  --model models/mobilenet_v3_small_fp32.tflite \
  --manifest data/imagenet/val_manifest.csv \
  --limit 5000

python scripts/eval/accuracy_eval.py \
  --model models/mobilenet_v3_small_int8.tflite \
  --manifest data/imagenet/val_manifest.csv \
  --limit 5000

# Step C: compare and generate report
python scripts/eval/compare_accuracy.py \
  --fp32 results/accuracy/mobilenet_v3_small_fp32_results.json \
  --int8 results/accuracy/mobilenet_v3_small_int8_results.json
```

---

## Project Structure

```
android-edge-ai-benchmark-lab/
├── app/src/main/kotlin/com/edgeai/benchmark/
│   ├── benchmark/        # BenchmarkEngine (abstract) + LiteRtEngine / OnnxEngine / ExecuTorchEngine
│   ├── model/            # BenchmarkResult data class, Runtime/Backend/Precision enums
│   ├── ui/               # MainActivity, ResultsAdapter
│   └── util/             # ThermalMonitor, MemoryTracker, CsvExporter
├── scripts/
│   ├── convert/          # PyTorch → ONNX / TFLite / ExecuTorch / TorchAO
│   ├── eval/             # Accuracy evaluation pipeline (prepare → eval → compare)
│   └── analyze/          # CSV → latency/memory charts (matplotlib / seaborn)
├── results/
│   ├── raw/              # Raw CSV from device (git-ignored)
│   ├── graphs/           # Generated charts (git-ignored)
│   └── accuracy/         # FP32 vs INT8 accuracy results and report
├── data/                 # ImageNet val dataset (git-ignored — large)
└── docs/                 # Screenshots, write-ups
```

---

## Device Matrix

| Device | SoC | Android | Results |
|---|---|---|---|
| Samsung Galaxy S26 Ultra (SM-S948N) | Snapdragon 8 Gen 3 | Android 16 | ✅ [v0.1 data](results/raw/) |

---

## Roadmap

- [x] **v0.1** — LiteRT (CPU + GPU), MobileNetV3-Small, p50/p95/p99, CSV export
- [x] **v0.2** — ONNX Runtime CPU, LiteRT vs ONNX Runtime comparison (3.8× gap found)
- [x] **v0.3** — ExecuTorch CPU, 3-runtime comparison (44× gap vs LiteRT found — XNNPACK backend required)
- [x] **v0.4** — ImageNet accuracy validation: FP32 69.4% baseline vs **full-integer INT8 PTQ collapse (0.6%)** — measured negative finding
- [ ] **v0.4.1** — FP16 (intermediate compression) + QAT recovery experiment
- [x] **v0.5** — YOLOv8n object detection (preprocess / inference / postprocess split)
  - [x] v0.5.0 — Python export (ONNX + TFLite) & cross-runtime correctness (class-score cosine 1.000)
  - [x] v0.5.1 — Android LiteRT inference: FP32 87.9 ms vs INT8 27.0 ms (INT8 3.3× faster on this heavy model)
  - [x] v0.5.2 — Android postprocess / NMS + sample visualization (Python↔Android parity: same detections, score Δ ≤ 0.006)
  - [x] v0.5.3 — Android 3-phase benchmark: INT8 inference 3.1× but end-to-end only 2.25× (quantize/dequant overhead in pre/post)
- [ ] **v1.0** — Multi-device matrix, technical blog series

---

## License

Apache License 2.0 — see [LICENSE](LICENSE)
