"""
dataset_explorer.py

Scans the ASL Alphabet dataset and the WLASL dataset and prints summary
statistics. Paths are never hardcoded - everything is resolved relative
to this script's location inside the project, so it works no matter
where the whole project folder is placed on disk (your laptop, a
teammate's laptop, a server, etc.).

Expected project layout (from the folder structure task):

    SignLanguagePlatform/
    |-- datasets/
    |   |-- asl_alphabet/
    |   |-- wlasl/
    |-- scripts/
        |-- dataset_explorer.py   <- this file

Usage:
    python dataset_explorer.py
    python dataset_explorer.py --datasets-dir /custom/path/to/datasets
"""

import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------
# Path resolution (no hardcoded paths)
# ---------------------------------------------------------------------

def get_default_datasets_dir() -> Path:
    """
    Resolve the datasets/ folder relative to this script's own location,
    not relative to the current working directory. This means the
    script works correctly whether you run it from inside scripts/,
    from the project root, or anywhere else.
    """
    script_dir = Path(__file__).resolve().parent          # .../scripts
    project_root = script_dir.parent                       # .../SignLanguagePlatform
    return project_root / "datasets"


def find_first_existing(candidates):
    """Return the first path in candidates that actually exists, or None."""
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_class_root(path: Path) -> Path:
    """
    Some dataset zips (this ASL Alphabet one included) extract with an
    extra wrapper folder, e.g.:

        asl_alphabet_train/
            asl_alphabet_train/    <- the real class folders are in here
                A/
                B/
                ...

    This walks downward through any folder that only contains a single
    subfolder and no files, until it reaches the level that actually
    holds the class folders (multiple subdirectories, or files).
    """
    current = path
    while True:
        entries = list(current.iterdir())
        subdirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]

        # Only descend if this folder is just a single wrapper folder
        # around one child folder with no files alongside it.
        if len(subdirs) == 1 and not files:
            current = subdirs[0]
            continue
        break

    return current


# ---------------------------------------------------------------------
# Task 1a: ASL Alphabet Explorer
# ---------------------------------------------------------------------

def explore_asl_alphabet(datasets_dir: Path):
    print("\n" + "=" * 60)
    print("ASL ALPHABET DATASET")
    print("=" * 60)

    asl_root = datasets_dir / "asl_alphabet"
    if not asl_root.exists():
        print(f"[!] Could not find folder: {asl_root}")
        return

    # The Kaggle ASL Alphabet dataset nests images inside
    # asl_alphabet_train/. Handle both that case and a flatter layout,
    # so the script doesn't break if someone reorganizes it slightly.
    image_root = find_first_existing([
        asl_root / "asl_alphabet_train",
        asl_root,
    ])

    if image_root is None:
        print(f"[!] No image folder found under: {asl_root}")
        return

    # Auto-descend through any wrapper folder(s) caused by how the
    # dataset zip was extracted (e.g. asl_alphabet_train/asl_alphabet_train/)
    resolved_root = resolve_class_root(image_root)
    if resolved_root != image_root:
        print(f"[i] Detected nested wrapper folder(s). Using: {resolved_root}")
    image_root = resolved_root

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    # Each subfolder of image_root is treated as one class.
    class_dirs = [d for d in image_root.iterdir() if d.is_dir()]

    if not class_dirs:
        print(f"[!] No class folders found inside: {image_root}")
        return

    class_counts = {}
    for class_dir in class_dirs:
        count = sum(
            1 for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        )
        class_counts[class_dir.name] = count

    # Print class names alphabetically, with their image counts.
    print(f"\nScanned folder: {image_root}\n")
    for class_name in sorted(class_counts.keys()):
        print(f"  {class_name:<12} -> {class_counts[class_name]} images")

    total_classes = len(class_counts)
    total_images = sum(class_counts.values())

    print(f"\nTotal classes : {total_classes}")
    print(f"Total images  : {total_images}")


# ---------------------------------------------------------------------
# Task 1b: WLASL Explorer
# ---------------------------------------------------------------------

def explore_wlasl(datasets_dir: Path):
    print("\n" + "=" * 60)
    print("WLASL DATASET")
    print("=" * 60)

    wlasl_root = datasets_dir / "wlasl"
    if not wlasl_root.exists():
        print(f"[!] Could not find folder: {wlasl_root}")
        return

    # Locate the annotation JSON file by searching for it, rather than
    # assuming an exact filename/location - some mirrors name or place
    # it slightly differently.
    json_candidates = list(wlasl_root.rglob("*.json"))

    if not json_candidates:
        print(f"[!] No annotation JSON file found inside: {wlasl_root}")
        return

    # Prefer a file that actually looks like the WLASL annotation file.
    annotation_file = next(
        (p for p in json_candidates if "wlasl" in p.name.lower()),
        json_candidates[0],
    )

    print(f"\nAnnotation file found: {annotation_file}")

    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # WLASL_v0.3.json is a list of entries, each with a "gloss" (the
    # word/label) and an "instances" list of video clips for that word.
    glosses = [entry.get("gloss") for entry in data if "gloss" in entry]
    unique_signs = sorted(set(glosses))

    print(f"\nTotal unique signs (glosses): {len(unique_signs)}")

    print("\nSample entries:")
    for entry in data[:5]:
        gloss = entry.get("gloss", "UNKNOWN")
        instances = entry.get("instances", [])
        num_instances = len(instances)
        sample_video_id = instances[0].get("video_id") if instances else "N/A"
        print(
            f"  gloss='{gloss}' | instances={num_instances} "
            f"| example video_id={sample_video_id}"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Explore ASL Alphabet and WLASL datasets."
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default=None,
        help=(
            "Path to the datasets/ folder. If not given, it is auto-"
            "detected relative to this script's location."
        ),
    )
    args = parser.parse_args()

    datasets_dir = (
        Path(args.datasets_dir).resolve()
        if args.datasets_dir
        else get_default_datasets_dir()
    )

    print(f"Using datasets directory: {datasets_dir}")

    if not datasets_dir.exists():
        print(f"[!] datasets directory does not exist: {datasets_dir}")
        return

    explore_asl_alphabet(datasets_dir)
    explore_wlasl(datasets_dir)


if __name__ == "__main__":
    main()