"""
dataset_explorer_v2.py

Weekend Task 7 - Dataset Explorer (upgraded)

Analyzes the ASL dataset and generates:
    - Total number of gesture classes
    - Total number of images
    - Number of images per class
    - Largest class
    - Smallest class
    - Exports the results into a CSV file

Paths are resolved relative to this script's location, same as
dataset_explorer.py - no hardcoded paths.
"""

import argparse
import csv
from pathlib import Path


def get_default_datasets_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    return project_root / "datasets"


def find_first_existing(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_class_root(path: Path) -> Path:
    """
    Auto-descend through any wrapper folder(s) caused by how a dataset
    zip was extracted (e.g. asl_alphabet_train/asl_alphabet_train/).
    """
    current = path
    while True:
        entries = list(current.iterdir())
        subdirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]

        if len(subdirs) == 1 and not files:
            current = subdirs[0]
            continue
        break

    return current


def analyze_asl_alphabet(datasets_dir: Path):
    asl_root = datasets_dir / "asl_alphabet"
    if not asl_root.exists():
        print(f"[!] Could not find folder: {asl_root}")
        return None

    image_root = find_first_existing([
        asl_root / "asl_alphabet_train",
        asl_root,
    ])

    if image_root is None:
        print(f"[!] No image folder found under: {asl_root}")
        return None

    image_root = resolve_class_root(image_root)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    class_dirs = [d for d in image_root.iterdir() if d.is_dir()]

    if not class_dirs:
        print(f"[!] No class folders found inside: {image_root}")
        return None

    class_counts = {}
    for class_dir in class_dirs:
        count = sum(
            1 for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        )
        class_counts[class_dir.name] = count

    return class_counts


def export_to_csv(class_counts: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "image_count"])
        for class_name in sorted(class_counts.keys()):
            writer.writerow([class_name, class_counts[class_name]])

    print(f"[i] CSV exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze the ASL Alphabet dataset and export a CSV report."
    )
    parser.add_argument("--datasets-dir", type=str, default=None)
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Path for the output CSV file (default: scripts/dataset_report.csv)",
    )
    args = parser.parse_args()

    datasets_dir = (
        Path(args.datasets_dir).resolve()
        if args.datasets_dir
        else get_default_datasets_dir()
    )

    print(f"Using datasets directory: {datasets_dir}")

    class_counts = analyze_asl_alphabet(datasets_dir)

    if not class_counts:
        return

    total_classes = len(class_counts)
    total_images = sum(class_counts.values())

    largest_class = max(class_counts, key=class_counts.get)
    smallest_class = min(class_counts, key=class_counts.get)

    print("\n" + "=" * 50)
    print("ASL ALPHABET DATASET REPORT")
    print("=" * 50)

    for class_name in sorted(class_counts.keys()):
        print(f"  {class_name:<10} -> {class_counts[class_name]} images")

    print(f"\nTotal classes  : {total_classes}")
    print(f"Total images   : {total_images}")
    print(f"Largest class  : {largest_class} ({class_counts[largest_class]} images)")
    print(f"Smallest class : {smallest_class} ({class_counts[smallest_class]} images)")

    output_csv = (
        Path(args.output_csv).resolve()
        if args.output_csv
        else Path(__file__).resolve().parent / "dataset_report.csv"
    )
    export_to_csv(class_counts, output_csv)


if __name__ == "__main__":
    main()