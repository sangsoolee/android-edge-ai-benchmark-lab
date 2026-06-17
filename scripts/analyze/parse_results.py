#!/usr/bin/env python3
"""
parse_results.py
----------------
Reads raw CSV exports from the Android benchmark app and prints a
per-(runtime × backend × precision) statistics summary.

Usage:
  python parse_results.py --input ../../results/raw/results/
  python parse_results.py --input ../../results/raw/results/ --markdown
"""

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(input_dir: Path) -> pd.DataFrame:
    csv_files = sorted(input_dir.rglob("benchmark_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No benchmark_*.csv files found under {input_dir}")

    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # latency_mode is a newer column; older CSVs predate it. Default to end_to_end
    # so grouping on it doesn't silently drop those rows (groupby drops NaN keys).
    if "latency_mode" not in df.columns:
        df["latency_mode"] = "end_to_end"
    else:
        df["latency_mode"] = df["latency_mode"].fillna("end_to_end")

    print(f"Loaded {len(df)} rows from {len(csv_files)} file(s)\n")
    return df


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

STATS_COLS = [
    "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
    "cold_start_ms", "peak_memory_mb", "model_size_mb",
]

GROUP_COLS = ["runtime", "backend", "model_name", "precision", "latency_mode"]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    # Keep most-recent run per configuration (same device session may repeat runs)
    df = df.sort_values("timestamp_utc_ms")
    deduped = df.groupby(GROUP_COLS).last().reset_index()

    summary = deduped[GROUP_COLS + STATS_COLS].copy()
    for col in STATS_COLS:
        summary[col] = summary[col].round(3)
    return summary.sort_values(["model_name", "p50_latency_ms"])


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------

def to_markdown(df: pd.DataFrame) -> str:
    cols = GROUP_COLS + STATS_COLS
    df = df[cols]
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise benchmark CSV results")
    parser.add_argument("--input",    type=Path, required=True,
                        help="Directory containing benchmark CSV files")
    parser.add_argument("--markdown", action="store_true",
                        help="Print output as a Markdown table")
    args = parser.parse_args()

    df = load(args.input)
    summary = aggregate(df)

    if args.markdown:
        print(to_markdown(summary))
    else:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 160)
        print(summary.to_string(index=False))

    print(f"\nTotal configurations: {len(summary)}")


if __name__ == "__main__":
    main()
