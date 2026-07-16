"""
extract_landmarks.py

Task 1 - Landmark Extraction Utility
Task 2 - Invalid Sample Handling
Task 3 - Dataset Statistics
Task 4 - (Optional) Coordinate Normalization

Walks every image in the ASL Alphabet dataset, runs MediaPipe hand
detection on each one, flattens the 21 landmarks (x, y, z) into a
63-value feature vector, and writes one row per successfully-detected
image to a CSV file:

    x0,y0,z0,x1,y1,z1,...,x20,y20,z20,label

This CSV becomes the training data for the gesture classifier.

Usage:
    python extract_landmarks.py
    python extract_landmarks.py --limit 200        (quick test: 200 images/class)
    python extract_landmarks.py --normalize         (also normalize coordinates)
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp


# ---------------------------------------------------------------------
# Path resolution (same pattern as dataset_explorer.py - no hardcoded paths)
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Landmark extraction + normalization
# ---------------------------------------------------------------------

def flatten_landmarks(hand_landmarks):
    """Turn MediaPipe's 21 (x, y, z) landmarks into a flat list of 63 values."""
    values = []
    for lm in hand_landmarks.landmark:
        values.extend([lm.x, lm.y, lm.z])
    return values


def normalize_landmarks(flat_values):
    """
    Task 4 (optional): make the landmarks less sensitive to the hand's
    absolute position/size in the frame.

    Two simple, commonly-used ideas combined here:
      1. Translation invariance: subtract the wrist (landmark 0)
         position from every point, so the wrist becomes the origin
         (0, 0, 0). This removes "where in the frame is the hand"
         as a factor.
      2. Scale invariance: divide every coordinate by the distance
         from the wrist to the middle finger MCP joint (landmark 9),
         a reasonably stable "hand size" reference. This removes
         "how close is the hand to the camera" as a factor.

    This is one straightforward approach, not the only one - the
    mentor mentioned we'll discuss normalization strategies later,
    so treat this as a baseline to compare against, not a final answer.
    """
    points = [flat_values[i:i + 3] for i in range(0, len(flat_values), 3)]

    wrist = points[0]
    middle_mcp = points[9]

    # Translate so wrist is the origin.
    translated = [
        [p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]]
        for p in points
    ]

    # Scale by wrist-to-middle-MCP distance (guard against division by ~0).
    scale = (
        (translated[9][0] ** 2 + translated[9][1] ** 2 + translated[9][2] ** 2) ** 0.5
    )
    if scale < 1e-6:
        scale = 1e-6

    normalized = [
        [p[0] / scale, p[1] / scale, p[2] / scale]
        for p in translated
    ]

    flat_normalized = []
    for p in normalized:
        flat_normalized.extend(p)

    return flat_normalized


# ---------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks from the ASL Alphabet dataset into a CSV."
    )
    parser.add_argument("--datasets-dir", type=str, default=None)
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Output CSV path (default: scripts/landmarks_dataset.csv)",
    )
    parser.add_argument(
        "--report", type=str, default=None,
        help="Output stats report path (default: scripts/extraction_report.json)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process this many images per class (useful for quick testing).",
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Also normalize landmarks (translation + scale invariant) and "
             "write them to a second CSV alongside the raw one.",
    )
    parser.add_argument(
        "--min-detection-confidence", type=float, default=0.3,
        help="Lower this if too many images are failing detection (default 0.3). "
             "Tightly-cropped hand images sometimes need a lower threshold.",
    )
    args = parser.parse_args()

    datasets_dir = (
        Path(args.datasets_dir).resolve()
        if args.datasets_dir
        else get_default_datasets_dir()
    )

    asl_root = datasets_dir / "asl_alphabet"
    if not asl_root.exists():
        print(f"[!] Could not find folder: {asl_root}")
        return

    image_root = find_first_existing([
        asl_root / "asl_alphabet_train",
        asl_root,
    ])
    if image_root is None:
        print(f"[!] No image folder found under: {asl_root}")
        return

    image_root = resolve_class_root(image_root)
    class_dirs = sorted([d for d in image_root.iterdir() if d.is_dir()])

    if not class_dirs:
        print(f"[!] No class folders found inside: {image_root}")
        return

    print(f"Using image folder: {image_root}")
    print(f"Found {len(class_dirs)} classes.")
    if args.limit:
        print(f"[i] --limit set: only processing up to {args.limit} images per class.")

    output_csv = (
        Path(args.output_csv).resolve()
        if args.output_csv
        else Path(__file__).resolve().parent / "landmarks_dataset.csv"
    )
    report_path = (
        Path(args.report).resolve()
        if args.report
        else Path(__file__).resolve().parent / "extraction_report.json"
    )

    normalized_csv = output_csv.with_name(output_csv.stem + "_normalized.csv")

    # MediaPipe Hands set up for STATIC images (not video), since we're
    # processing independent, unrelated photos - not a continuous stream.
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
    )

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]

    total_processed = 0
    total_success = 0
    total_failed = 0
    per_class_counts = {}
    skipped_files = []

    start_time = time.time()

    with open(output_csv, "w", newline="", encoding="utf-8") as raw_f, \
         (open(normalized_csv, "w", newline="", encoding="utf-8") if args.normalize else open(__import__("os").devnull, "w")) as norm_f:

        raw_writer = csv.writer(raw_f)
        raw_writer.writerow(header)

        norm_writer = None
        if args.normalize:
            norm_writer = csv.writer(norm_f)
            norm_writer.writerow(header)

        for class_dir in class_dirs:
            class_name = class_dir.name
            per_class_counts[class_name] = 0

            image_files = [
                f for f in sorted(class_dir.iterdir())
                if f.is_file() and f.suffix.lower() in valid_extensions
            ]

            if args.limit:
                image_files = image_files[:args.limit]

            for image_path in image_files:
                total_processed += 1

                image = cv2.imread(str(image_path))
                if image is None:
                    total_failed += 1
                    skipped_files.append(str(image_path))
                    continue

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_image)

                if not results.multi_hand_landmarks:
                    total_failed += 1
                    skipped_files.append(str(image_path))
                    continue

                hand_landmarks = results.multi_hand_landmarks[0]
                flat_values = flatten_landmarks(hand_landmarks)

                raw_writer.writerow(flat_values + [class_name])

                if args.normalize:
                    normalized_values = normalize_landmarks(flat_values)
                    norm_writer.writerow(normalized_values + [class_name])

                total_success += 1
                per_class_counts[class_name] += 1

                if total_processed % 500 == 0:
                    elapsed = time.time() - start_time
                    print(f"  ...processed {total_processed} images "
                          f"({total_success} ok, {total_failed} failed) "
                          f"- {elapsed:.1f}s elapsed")

    hands.close()

    elapsed_total = time.time() - start_time

    # --- Task 2: log skipped files ---
    skipped_log_path = Path(__file__).resolve().parent / "skipped_images.log"
    with open(skipped_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(skipped_files))

    # --- Task 3: summary + report ---
    print("\n" + "=" * 55)
    print("EXTRACTION SUMMARY")
    print("=" * 55)
    print(f"Total images processed        : {total_processed}")
    print(f"Successful landmark extractions: {total_success}")
    print(f"Failed landmark detections     : {total_failed}")
    print(f"Number of classes               : {len(per_class_counts)}")
    print(f"Time elapsed                    : {elapsed_total:.1f}s")
    print("\nSamples per class:")
    for class_name in sorted(per_class_counts.keys()):
        print(f"  {class_name:<10} -> {per_class_counts[class_name]} samples")

    print(f"\n[i] Raw landmark CSV     : {output_csv}")
    if args.normalize:
        print(f"[i] Normalized CSV       : {normalized_csv}")
    print(f"[i] Skipped files log    : {skipped_log_path}")

    report = {
        "total_images_processed": total_processed,
        "successful_extractions": total_success,
        "failed_detections": total_failed,
        "num_classes": len(per_class_counts),
        "samples_per_class": per_class_counts,
        "elapsed_seconds": round(elapsed_total, 2),
        "min_detection_confidence_used": args.min_detection_confidence,
        "normalized_output_generated": bool(args.normalize),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[i] Stats report saved to: {report_path}")


if __name__ == "__main__":
    main()