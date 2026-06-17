#!/usr/bin/env python3
"""
download_imagenet_val.py
------------------------
Downloads a subset of ImageNet validation images from the Kaggle competition
"imagenet-object-localization-challenge", then writes val_manifest.csv.

Why not the `kaggle` CLI?
  The kaggle 2.x CLI (kagglesdk gRPC endpoint) returns 404 for deeply-nested
  competition file paths like ILSVRC/Data/CLS-LOC/val/*.JPEG. The legacy REST
  endpoint /api/v1/competitions/data/download/{comp}/{path} still works, so we
  call it directly with HTTP Basic Auth read from ~/.kaggle/kaggle.json.

Setup (one-time):
  1. ~/.kaggle/kaggle.json must exist:  {"username":"...","key":"..."}
  2. Accept competition rules once in the browser:
     https://www.kaggle.com/competitions/imagenet-object-localization-challenge/rules

Usage:
  python scripts/eval/download_imagenet_val.py --n 500
  python scripts/eval/download_imagenet_val.py --n 5000 --workers 16

Output:
  data/imagenet/val/<image_id>.JPEG
  data/imagenet/val_manifest.csv   (image_path,label_index,wnid,class_name)
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

COMPETITION  = "imagenet-object-localization-challenge"
VAL_PREFIX   = "ILSVRC/Data/CLS-LOC/val"
BASE_URL     = "https://www.kaggle.com/api/v1/competitions/data/download"

REPO_ROOT    = Path(__file__).parents[2]
DATA_DIR     = REPO_ROOT / "data" / "imagenet"
SOLUTION_CSV = DATA_DIR / "LOC_val_solution.csv"
OUT_IMAGES   = DATA_DIR / "val"
OUT_MANIFEST = DATA_DIR / "val_manifest.csv"

JPEG_MAGIC = b"\xff\xd8\xff"


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def load_kaggle_credentials() -> tuple[str, str]:
    """Read username/key from ~/.kaggle/kaggle.json (or KAGGLE_* env vars)."""
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key  = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        return env_user, env_key

    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if not cred_path.exists():
        raise SystemExit(
            f"No credentials. Create {cred_path} with "
            '{"username":"...","key":"..."} or set KAGGLE_USERNAME/KAGGLE_KEY.'
        )
    with open(cred_path) as f:
        data = json.load(f)
    return data["username"], data["key"]


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def load_wnid_to_idx_and_name() -> tuple[dict[str, int], dict[str, str]]:
    """Build wnid → 0-based ImageNet index.

    Key fact: the ImageNet-1k class index used by torchvision IS the set of 1000
    wnids sorted in ascending (alphanumeric) order — imagenet_class_index.json
    is exactly {0: n01440764, 1: n01443537, ...}.  So we don't need any bundled
    file: collect the 1000 unique wnids from the FULL LOC_val_solution.csv, sort
    them, and enumerate.  This matches the model's output ordering exactly.

    class_name is cosmetic (manifest only) — left as the wnid if no names file.
    """
    if not SOLUTION_CSV.exists():
        raise SystemExit(f"{SOLUTION_CSV} not found — download LOC_val_solution.csv first.")

    wnids = set()
    with open(SOLUTION_CSV, newline="") as f:
        for row in csv.DictReader(f):
            wnids.add(row["PredictionString"].strip().split()[0])

    if len(wnids) != 1000:
        print(f"  ⚠️  Expected 1000 unique wnids, found {len(wnids)} — "
              "label indices may be off if the CSV is partial.")

    sorted_wnids = sorted(wnids)
    wnid_to_idx  = {wnid: idx for idx, wnid in enumerate(sorted_wnids)}

    # Optional: attach human-readable names if a class-index JSON happens to exist
    wnid_to_name = {wnid: wnid for wnid in sorted_wnids}
    names = _maybe_load_names()
    if names:
        for wnid in sorted_wnids:
            wnid_to_name[wnid] = names.get(wnid, wnid)

    print(f"  ✅  Built wnid→idx mapping for {len(wnid_to_idx)} classes "
          "(sorted-wnid ordering = torchvision index)")
    return wnid_to_idx, wnid_to_name


def _maybe_load_names() -> dict[str, str]:
    """Best-effort wnid → class-name from a bundled imagenet_class_index.json."""
    try:
        import torchvision
        base = Path(torchvision.__file__).parent
        hits = list(base.rglob("imagenet_class_index.json"))
        if hits:
            with open(hits[0]) as f:
                data = json.load(f)
            return {v[0]: v[1] for v in data.values()}
    except Exception:
        pass
    return {}


def read_gt(n: int) -> list[tuple[str, str]]:
    """Return first n (image_id, wnid) pairs from LOC_val_solution.csv."""
    if not SOLUTION_CSV.exists():
        raise SystemExit(
            f"{SOLUTION_CSV} not found.\n"
            "Download it first:\n"
            "  kaggle competitions download -c "
            f"{COMPETITION} -f LOC_val_solution.csv -p {DATA_DIR}\n"
            f"  unzip {DATA_DIR}/LOC_val_solution.csv.zip -d {DATA_DIR}"
        )
    pairs = []
    with open(SOLUTION_CSV, newline="") as f:
        for row in csv.DictReader(f):
            image_id = row["ImageId"].strip()
            wnid     = row["PredictionString"].strip().split()[0]
            pairs.append((image_id, wnid))
            if len(pairs) >= n:
                break
    return pairs


# ---------------------------------------------------------------------------
# Download (via kaggle CLI — authenticates with the new KGAT token; the legacy
# REST endpoint rejects KGAT tokens with 401, so we shell out to the CLI which
# downloads individual nested competition files just fine.)
# ---------------------------------------------------------------------------

def download_one(image_id: str) -> tuple[str, bool, str]:
    import subprocess, zipfile

    out_path = OUT_IMAGES / f"{image_id}.JPEG"
    if out_path.exists() and out_path.stat().st_size > 0:
        return image_id, True, "exists"

    remote = f"{VAL_PREFIX}/{image_id}.JPEG"
    proc = subprocess.run(
        ["kaggle", "competitions", "download",
         "-c", COMPETITION, "-f", remote, "-p", str(OUT_IMAGES)],
        capture_output=True, text=True,
    )

    # CLI may save the file directly, or as <image_id>.JPEG.zip — handle both.
    if out_path.exists() and out_path.stat().st_size > 0:
        return image_id, True, "ok"

    zip_path = OUT_IMAGES / f"{image_id}.JPEG.zip"
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(OUT_IMAGES)
            zip_path.unlink(missing_ok=True)
        except Exception as e:
            return image_id, False, f"unzip:{e.__class__.__name__}"
        if out_path.exists() and out_path.stat().st_size > 0:
            return image_id, True, "ok-zip"

    err = (proc.stderr or proc.stdout or "").strip().splitlines()
    reason = err[-1][:60] if err else f"rc={proc.returncode}"
    return image_id, False, reason


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download ImageNet val images from Kaggle")
    parser.add_argument("--n", type=int, default=500,
                        help="Number of images to download (default: 500)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download threads (default: 8)")
    args = parser.parse_args()

    # Credentials are used by the kaggle CLI itself; just verify they exist.
    load_kaggle_credentials()
    wnid_to_idx, wnid_to_name = load_wnid_to_idx_and_name()

    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    pairs = read_gt(args.n)
    print(f"Downloading {len(pairs)} val images to {OUT_IMAGES}  (workers={args.workers})\n")

    ok = fail = 0
    failures: list[str] = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, img_id): img_id
                   for img_id, _ in pairs}
        for i, fut in enumerate(as_completed(futures), 1):
            image_id, success, reason = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                failures.append(f"{image_id} ({reason})")

            if i % 50 == 0 or i == len(pairs):
                elapsed = time.perf_counter() - t0
                rate = i / elapsed if elapsed else 0
                eta  = (len(pairs) - i) / rate if rate else 0
                print(f"  [{i}/{len(pairs)}]  ok={ok}  fail={fail}  "
                      f"{rate:.1f} img/s  ETA {eta:.0f}s")

    # Write manifest from successfully-downloaded files
    wnid_by_id = dict(pairs)
    written = skipped = 0
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label_index", "wnid", "class_name"])
        for image_id, wnid in pairs:
            img_path = OUT_IMAGES / f"{image_id}.JPEG"
            if not (img_path.exists() and img_path.stat().st_size > 0):
                skipped += 1
                continue
            label_idx = wnid_to_idx.get(wnid)
            if label_idx is None:
                skipped += 1
                continue
            writer.writerow([str(img_path.resolve()), label_idx, wnid,
                             wnid_to_name.get(wnid, "unknown")])
            written += 1

    elapsed = time.perf_counter() - t0
    print(f"\n{'='*56}")
    print(f"  Downloaded : {ok}   Failed: {fail}   Time: {elapsed:.0f}s")
    print(f"  Manifest   : {OUT_MANIFEST}  ({written} rows, {skipped} skipped)")
    print(f"{'='*56}")

    if failures[:5]:
        print("\n  First failures:")
        for line in failures[:5]:
            print(f"    {line}")
        print("\n  If most failures mention 401/403, accept the competition rules once:")
        print(f"     https://www.kaggle.com/competitions/{COMPETITION}/rules")
        print("  Re-run to retry failures (existing files are skipped).")
    print()


if __name__ == "__main__":
    main()
