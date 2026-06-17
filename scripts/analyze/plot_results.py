#!/usr/bin/env python3
"""
plot_results.py
---------------
Reads CSV benchmark exports from the Android app and generates comparison charts.

Usage:
  python plot_results.py --input ../../results/raw/ --output ../../results/graphs/

Outputs:
  latency_comparison.png   — p50/p95/p99 bar chart per runtime × backend
  latency_cdf.png          — latency distribution (if raw per-run data available)
  memory_comparison.png    — peak memory bar chart
  cold_start.png           — cold-start time comparison
  summary_table.csv        — aggregated comparison table
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

RUNTIME_COLORS = {
    "LiteRT":       "#4285F4",   # Google Blue
    "ONNXRuntime":  "#00A4EF",   # Microsoft Blue
    "ExecuTorch":   "#EE4B2B",   # PyTorch Red
}

PRECISION_HATCHES = {
    "FP32": "",
    "FP16": "//",
    "INT8": "xx",
}

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_results(input_dir: Path) -> pd.DataFrame:
    csv_files = sorted(input_dir.glob("benchmark_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No benchmark CSV files found in {input_dir}")

    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    df["label"] = df["runtime"] + "\n" + df["backend"] + "\n" + df["precision"]
    print(f"Loaded {len(df)} result rows from {len(csv_files)} file(s)")
    return df

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_latency_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))

    x = range(len(df))
    width = 0.25

    bars_p50 = ax.bar([i - width for i in x], df["p50_latency_ms"], width,
                      label="p50", color="#4CAF50", alpha=0.85)
    bars_p95 = ax.bar(x,                       df["p95_latency_ms"], width,
                      label="p95", color="#FF9800", alpha=0.85)
    bars_p99 = ax.bar([i + width for i in x],  df["p99_latency_ms"], width,
                      label="p99", color="#F44336", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=9)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency by Runtime × Backend × Precision")
    ax.legend()
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    out = output_dir / "latency_comparison.png"
    plt.savefig(out)
    plt.close()
    print(f"  → {out}")


def plot_memory_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df, x="label", y="peak_memory_mb",
                palette="Blues_d", ax=ax)
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_xlabel("")
    ax.set_title("Peak PSS Memory by Runtime × Backend × Precision")
    plt.xticks(fontsize=9)
    plt.tight_layout()
    out = output_dir / "memory_comparison.png"
    plt.savefig(out)
    plt.close()
    print(f"  → {out}")


def plot_cold_start(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=df, x="label", y="cold_start_ms",
                palette="Oranges_d", ax=ax)
    ax.set_ylabel("Cold Start (ms)")
    ax.set_xlabel("")
    ax.set_title("Model Load + First Inference (Cold Start)")
    plt.xticks(fontsize=9)
    plt.tight_layout()
    out = output_dir / "cold_start.png"
    plt.savefig(out)
    plt.close()
    print(f"  → {out}")


def generate_summary_table(df: pd.DataFrame, output_dir: Path) -> None:
    cols = [
        "runtime", "backend", "model_name", "precision",
        "model_size_mb",
        "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
        "cold_start_ms", "peak_memory_mb",
        "thermal_before", "thermal_after",
        "device_model", "android_version"
    ]
    summary = df[[c for c in cols if c in df.columns]].copy()
    # Round numeric columns for readability
    for col in summary.select_dtypes(include="float").columns:
        summary[col] = summary[col].round(2)
    out = output_dir / "summary_table.csv"
    summary.to_csv(out, index=False)
    print(f"  → {out}")
    print("\n" + summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot Android Edge AI benchmark results")
    parser.add_argument("--input",  required=True, type=Path, help="Directory containing benchmark CSV files")
    parser.add_argument("--output", required=True, type=Path, help="Directory for output charts")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    df = load_results(args.input)

    print("\nGenerating charts...")
    plot_latency_comparison(df, args.output)
    plot_memory_comparison(df, args.output)
    plot_cold_start(df, args.output)
    generate_summary_table(df, args.output)

    print(f"\n✅  All charts saved to {args.output}\n")


if __name__ == "__main__":
    main()
