#!/usr/bin/env python3
"""
prepare_imagenet_val.py
-----------------------
Extracts ImageNet validation images and builds a manifest CSV for accuracy_eval.py.

Kaggle dataset: imagenet-object-localization-challenge
  Download: kaggle competitions download -c imagenet-object-localization-challenge
  Contents after extraction:
    LOC_val_solution.csv          ← ground truth (wnid per image)
    LOC_synset_mapping.txt        ← wnid → class name
    ILSVRC/Data/CLS-LOC/val/      ← 50,000 validation JPEGs (flat structure)

Alternative: if you have ILSVRC2012_validation_ground_truth.txt (1-based integer labels),
             use --gt-txt instead of --solution-csv.

Usage (Kaggle layout):
  python prepare_imagenet_val.py \\
    --val-images-dir /path/to/ILSVRC/Data/CLS-LOC/val \\
    --solution-csv   /path/to/LOC_val_solution.csv \\
    --output-dir     data/imagenet

Usage (txt label layout):
  python prepare_imagenet_val.py \\
    --val-images-dir /path/to/val \\
    --gt-txt         /path/to/ILSVRC2012_validation_ground_truth.txt \\
    --output-dir     data/imagenet

Output:
  data/imagenet/val_manifest.csv  — columns: image_path,label_index,wnid,class_name
    image_path  : absolute path to JPEG
    label_index : 0-based ImageNet class index matching model output
    wnid        : WordNet synset ID (e.g. n01440764)
    class_name  : human-readable name (e.g. tench)
"""

import argparse
import csv
import json
from pathlib import Path

import torchvision  # only for imagenet_class_index.json path

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def _find_imagenet_class_index_json() -> Path:
    """Locate imagenet_class_index.json, downloading from GitHub if needed."""
    import urllib.request
    base = Path(torchvision.__file__).parent
    candidate = base / "data" / "imagenet_class_index.json"
    if not candidate.exists():
        hits = list(base.rglob("imagenet_class_index.json"))
        if hits:
            candidate = hits[0]
    if not candidate.exists():
        cache = Path.home() / ".cache" / "imagenet_class_index.json"
        if not cache.exists():
            url = ("https://raw.githubusercontent.com/pytorch/vision/main/"
                   "torchvision/data/imagenet_class_index.json")
            print("  imagenet_class_index.json not found locally — downloading from GitHub...")
            cache.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, cache)
        candidate = cache
    return candidate


def load_torchvision_wnid_to_idx() -> dict[str, int]:
    with open(_find_imagenet_class_index_json()) as f:
        data = json.load(f)
    return {v[0]: int(k) for k, v in data.items()}


def load_torchvision_wnid_to_name() -> dict[str, str]:
    with open(_find_imagenet_class_index_json()) as f:
        data = json.load(f)
    return {v[0]: v[1] for v in data.values()}


# ---------------------------------------------------------------------------
# Ground truth parsers
# ---------------------------------------------------------------------------

def parse_solution_csv(solution_csv: Path) -> dict[str, str]:
    """Parse LOC_val_solution.csv → {image_id: wnid}.

    CSV format:
      ImageId,PredictionString
      ILSVRC2012_val_00000001,n01751748 156 48 272 332 n01751748 42 9 322 285
    The first token of PredictionString is the class wnid.
    """
    result = {}
    with open(solution_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["ImageId"].strip()
            wnid = row["PredictionString"].strip().split()[0]
            result[image_id] = wnid
    return result


def parse_gt_txt(gt_txt: Path, val_images: list[Path]) -> dict[str, str]:
    """Parse ILSVRC2012_validation_ground_truth.txt (1-based) → {image_id: wnid}.

    The txt file has 50,000 lines, one integer per line.
    Images must be sorted in the standard ILSVRC order to align correctly.

    The 1-based index maps to the synset via ILSVRC2012_mapping which is not
    included here — instead we use the sorted synset list from torchvision.
    """
    # torchvision's synset list is sorted 0-999 by wnid alphabetically
    json_path = Path(torchvision.__file__).parent / "data" / "imagenet_class_index.json"
    with open(json_path) as f:
        class_idx = json.load(f)

    # ILSVRC2012 uses a different synset ordering (devkit wnids.txt).
    # Without that file we cannot reliably parse gt.txt — raise clearly.
    raise NotImplementedError(
        "--gt-txt requires ILSVRC2012_devkit_t12 synset ordering to map 1-based "
        "integer labels to wnids. Use --solution-csv (LOC_val_solution.csv) instead."
    )


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_manifest(
    val_images_dir: Path,
    gt_map: dict[str, str],           # {image_id (no ext): wnid}
    wnid_to_idx: dict[str, int],
    wnid_to_name: dict[str, str],
    output_csv: Path,
) -> int:
    jpegs = sorted(val_images_dir.glob("*.JPEG"))
    if not jpegs:
        jpegs = sorted(val_images_dir.glob("*.jpg"))
    if not jpegs:
        raise FileNotFoundError(f"No JPEG images found in {val_images_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label_index", "wnid", "class_name"])

        for jpeg in jpegs:
            image_id = jpeg.stem  # ILSVRC2012_val_00000001
            wnid = gt_map.get(image_id)
            if wnid is None:
                skipped += 1
                continue
            label_index = wnid_to_idx.get(wnid)
            if label_index is None:
                skipped += 1
                continue
            class_name = wnid_to_name.get(wnid, "unknown")
            writer.writerow([str(jpeg.resolve()), label_index, wnid, class_name])
            written += 1

    return written, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_kaggle_root(root: Path) -> tuple[Path, Path]:
    """Auto-discover val images dir and solution CSV under a Kaggle extraction root.

    Kaggle layout after extraction:
      <root>/
        LOC_val_solution.csv
        ILSVRC/Data/CLS-LOC/val/   ← 50k JPEGs
    """
    solution = root / "LOC_val_solution.csv"
    if not solution.exists():
        raise SystemExit(
            f"LOC_val_solution.csv not found under {root}\n"
            "Expected Kaggle layout: <root>/LOC_val_solution.csv"
        )

    # Try canonical Kaggle path first, then fallback glob
    val_dir = root / "ILSVRC" / "Data" / "CLS-LOC" / "val"
    if not val_dir.exists():
        candidates = list(root.rglob("val"))
        candidates = [p for p in candidates if p.is_dir() and any(p.glob("*.JPEG"))]
        if not candidates:
            raise SystemExit(
                f"Could not find validation image directory under {root}\n"
                "Expected: ILSVRC/Data/CLS-LOC/val/"
            )
        val_dir = candidates[0]
        print(f"  Auto-discovered val dir: {val_dir}")

    return val_dir, solution


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ImageNet val manifest from Kaggle dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover from Kaggle extraction root (recommended)
  python prepare_imagenet_val.py --imagenet-root data/imagenet

  # Manual paths
  python prepare_imagenet_val.py \\
    --val-images-dir data/imagenet/ILSVRC/Data/CLS-LOC/val \\
    --solution-csv   data/imagenet/LOC_val_solution.csv
""",
    )
    parser.add_argument("--imagenet-root", type=Path, default=None,
                        help="Kaggle extraction root (auto-discovers val dir + solution CSV)")
    parser.add_argument("--val-images-dir", type=Path, default=None,
                        help="Directory containing validation JPEGs")
    parser.add_argument("--solution-csv", type=Path, default=None,
                        help="Path to LOC_val_solution.csv (Kaggle format)")
    parser.add_argument("--gt-txt", type=Path, default=None,
                        help="Path to ILSVRC2012_validation_ground_truth.txt (1-based)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parents[2] / "data" / "imagenet",
                        help="Output directory (default: data/imagenet)")
    args = parser.parse_args()

    # Resolve paths from --imagenet-root shortcut
    if args.imagenet_root:
        args.val_images_dir, args.solution_csv = _resolve_kaggle_root(args.imagenet_root)

    if not args.val_images_dir:
        raise SystemExit("Provide --imagenet-root OR --val-images-dir + --solution-csv")
    if not args.solution_csv and not args.gt_txt:
        raise SystemExit("Provide --imagenet-root OR --solution-csv OR --gt-txt")

    print(f"\n[1/3] Loading ImageNet class index from torchvision...")
    wnid_to_idx  = load_torchvision_wnid_to_idx()
    wnid_to_name = load_torchvision_wnid_to_name()
    print(f"  → {len(wnid_to_idx)} classes loaded")

    print(f"[2/3] Parsing ground truth labels...")
    if args.solution_csv:
        gt_map = parse_solution_csv(args.solution_csv)
    else:
        gt_map = parse_gt_txt(args.gt_txt, [])
    print(f"  → {len(gt_map)} label entries")

    print(f"[3/3] Building manifest from {args.val_images_dir}...")
    output_csv = args.output_dir / "val_manifest.csv"
    written, skipped = build_manifest(
        args.val_images_dir, gt_map, wnid_to_idx, wnid_to_name, output_csv
    )

    print(f"\n✅  Done")
    print(f"   Manifest : {output_csv}")
    print(f"   Written  : {written:,} images")
    if skipped:
        print(f"   Skipped  : {skipped} (no label match)\n")


if __name__ == "__main__":
    main()
