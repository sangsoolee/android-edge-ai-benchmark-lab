# Android AI Benchmark — On-Device Inference Runtime Comparison

> **Does GPU always beat CPU? Does INT8 always beat FP32?**  
> We measured it on a real device. The answers might surprise you.

[![Android](https://img.shields.io/badge/Platform-Android%208.0+-green.svg)](https://developer.android.com)
[![Kotlin](https://img.shields.io/badge/Language-Kotlin-purple.svg)](https://kotlinlang.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Build](https://img.shields.io/github/actions/workflow/status/sangsoolee/android-edge-ai-benchmark-lab/build.yml?label=APK%20build)](https://github.com/sangsoolee/android-edge-ai-benchmark-lab/actions)

---

<p align="center">
  <img src="docs/screenshot_benchmark.png" width="320" alt="Benchmark results on Galaxy S26 Ultra"/>
</p>

---

## What We Found (Galaxy S26 Ultra · Snapdragon 8 Gen 3)

### MobileNetV3-Small · LiteRT

| Backend | Precision | p50 | p95 | p99 | Cold Start | Memory |
|---|---|---:|---:|---:|---:|---:|
| **CPU (XNNPACK)** | **FP32** | **1.42 ms** ✅ | **1.49 ms** | **1.59 ms** | 22.2 ms | 93.5 MB |
| CPU (XNNPACK) | INT8 | 2.86 ms | 3.06 ms | 3.08 ms | 15.5 ms | 100.0 MB |
| GPU delegate | FP32 | 2.83 ms | 3.11 ms | 3.26 ms | 204.8 ms | 163.2 MB |

> Measurement: warmup=20, measured=100 runs · release build · airplane mode · no charging · 5-min cooldown

**Key findings:**
- 🏆 **CPU (FP32) wins** — Snapdragon 8 Gen 3 + XNNPACK is so fast that GPU transfer overhead dominates for small models
- ⚠️ **GPU cold-start = 204 ms** — 9× slower than CPU (shader compilation cost). Never use GPU delegate for first-launch UX
- ⚠️ **INT8 is 2× slower than FP32** — dequantize ops added by TFLite converter cancel out the compute savings on this chip
- 📦 **INT8 is 3.5× smaller** (2.76 MB vs 9.73 MB) — the only win for INT8 is storage/download size

*The conventional wisdom that "GPU is always faster" and "INT8 is always faster" does not hold for lightweight models on modern mobile SoCs. Always benchmark on your target device.*

---

## What This Project Is

A **reproducible benchmarking pipeline** for comparing on-device AI inference runtimes on real Android hardware.

Most "on-device AI" articles benchmark on a simulator, use a single average latency number, or don't disclose measurement conditions. This project measures **p50/p95/p99 latency, cold-start time, peak PSS memory, and thermal status** under fully controlled conditions — and makes all raw CSV data available.

**Runtimes compared:**

| Runtime | Status | Backends |
|---|---|---|
| [LiteRT / TFLite](https://ai.google.dev/edge/litert) 2.x | ✅ v0.1 | CPU (XNNPACK), GPU delegate |
| [ONNX Runtime Android](https://onnxruntime.ai) 1.x | 🔜 v0.2 | CPU, NNAPI |
| [ExecuTorch](https://pytorch.org/executorch) | 🔜 v0.3 | CPU, XNNPACK |

---

## Reproducibility First

Every result in this repo can be reproduced by anyone with the same device. Measurement conditions are not just documented — they are enforced in code.

```
Warmup runs:      20 (discarded)
Measured runs:    100
Input:            Fixed synthetic tensor, same seed every run
Build type:       Release (never debug — interpreter overhead is significant)
ABI:              arm64-v8a only
Network:          Airplane mode ON
Screen:           50% brightness, fixed
Charging:         Cable disconnected
Cooldown:         5 min idle before each session
```

### Why p99?

p50 looks great on a chart. p99 is what your users actually experience on a bad thermal day.  
A runtime with `p50=12ms` but `p99=180ms` is not production-ready.

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
adb pull /sdcard/Android/data/com.edgeai.benchmark/files/results/ ./results/raw/
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
- [ ] **v0.2** — ONNX Runtime, LiteRT vs ONNX comparison charts
- [ ] **v0.3** — ExecuTorch
- [ ] **v0.4** — INT8 accuracy drop analysis (FP32 vs INT8 top-1 accuracy)
- [ ] **v0.5** — YOLOv8n / YOLOv11n (preprocess / inference / postprocess split)
- [ ] **v1.0** — Multi-device matrix, technical blog series

---

## License

Apache License 2.0 — see [LICENSE](LICENSE)
