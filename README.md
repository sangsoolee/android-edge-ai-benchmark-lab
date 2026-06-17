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

**Key findings:**
- 🏆 **LiteRT CPU FP32 is fastest (1.53 ms)** — Snapdragon 8 Gen 3 + XNNPACK. All three runtimes produce identical outputs (cosine ≈ 1.000, same top-1), so the gaps are pure runtime/backend efficiency — not divergent conversions.
- ⚡ **ONNX NNAPI beats ONNX CPU by ~40% at the median (5.43 → 3.28 ms)** — enabling the on-device accelerator helps, but the distribution is bimodal: NNAPI partially falls back to CPU for MobileNetV3's hard-swish ops, so p95/p99 stay near the CPU path.
- 🐢 **ExecuTorch XNNPACK lowering did NOT help here (62.6 → 74.1 ms)** — on the prebuilt `executorch-android:1.3.1` AAR we could not confirm the XNNPACK delegate engaged (no delegate logs, no speedup, measured under back-to-back thermal load). Lowering the `.pte` is necessary but not sufficient — production use needs a runtime build with the XNNPACK backend verified.
- ⚠️ **GPU cold start ≈ 177 ms** — ~10× the CPU path (shader compilation). Critical for first-launch UX.
- ⚠️ **INT8 is ~2× slower than FP32 on CPU (1.53 → 2.86 ms)** — dequantize ops outweigh compute savings on this chip; INT8's only win here is 3.5× smaller on disk.

*The "fair tier" lesson: enabling each runtime's accelerator (NNAPI, XNNPACK) did **not** universally close the gap — it helped ONNX modestly and didn't help ExecuTorch at all on this build. "GPU is faster", "INT8 is faster", "this accelerator is faster" are not universal truths — always benchmark on your target device.*

---

## What This Project Is

A **reproducible benchmarking pipeline** for comparing on-device AI inference runtimes on real Android hardware.

Most "on-device AI" articles benchmark on a simulator, use a single average latency number, or don't disclose measurement conditions. This project measures **p50/p95/p99 latency, cold-start time, peak PSS memory, and thermal status**, documents the exact measurement protocol, and commits the raw result rows — every benchmark run, including repeat sessions — to [results/raw/](results/raw/). (Each row aggregates one 100-inference run; committing the full per-inference distributions is planned.)

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
python scripts/convert/export_tflite.py --model mobilenet_v3_small --precision fp32
python scripts/convert/export_tflite.py --model mobilenet_v3_small --precision int8
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
python scripts/analyze/plot_results.py \
  --input results/raw/ \
  --output results/graphs/
```

---

## Project Structure

```
android-edge-ai-benchmark-lab/
├── app/src/main/kotlin/com/edgeai/benchmark/
│   ├── benchmark/        # BenchmarkEngine (abstract) + LiteRtEngine
│   ├── model/            # BenchmarkResult data class, Runtime/Backend/Precision enums
│   ├── ui/               # MainActivity, ResultsAdapter
│   └── util/             # ThermalMonitor, MemoryTracker, CsvExporter
├── scripts/
│   ├── convert/          # PyTorch → ONNX / TFLite / ExecuTorch
│   └── analyze/          # CSV → charts (matplotlib / seaborn)
├── results/
│   ├── raw/              # Raw CSV from device (git-ignored)
│   └── graphs/           # Generated charts (git-ignored)
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
- [ ] **v0.4** — INT8 accuracy drop analysis (FP32 vs INT8 top-1 accuracy)
- [ ] **v0.5** — YOLOv8n / YOLOv11n (preprocess / inference / postprocess split)
- [ ] **v1.0** — Multi-device matrix, technical blog series

---

## License

Apache License 2.0 — see [LICENSE](LICENSE)
