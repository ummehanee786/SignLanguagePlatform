"""
dataset_builder.py

Task 2: consumes extract_landmarks' generator, writes every successful
sample to a CSV file (the single source of truth for model training),
and collects counts along the way so validator.py doesn't need to
re-process every image just to generate its report.
"""

import csv
from pathlib import Path
from typing import Optional

from app.ai.preprocessing.extract_landmarks import extract_dataset_landmarks, normalize_landmarks


def build_column_names() -> list[str]:
    return [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]


def build_dataset(
    image_root: Path,
    output_csv_path: Path,
    limit_per_class: Optional[int] = None,
    normalize: bool = False,
    normalized_csv_path: Optional[Path] = None,
) -> dict:
    """
    Runs extraction over the whole dataset and writes successful
    samples to output_csv_path (and optionally a second, normalized
    CSV). Returns a stats dict that validator.py turns into the
    quality report - no need to scan the images twice.
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_images": 0,
        "successful": 0,
        "no_hand_detected": 0,
        "unreadable": 0,
        "unreadable_files": [],
        "no_hand_files": [],
    }

    header = build_column_names()

    norm_path = normalized_csv_path or output_csv_path.with_name(
        output_csv_path.stem + "_normalized.csv"
    )

    raw_f = open(output_csv_path, "w", newline="", encoding="utf-8")
    raw_writer = csv.writer(raw_f)
    raw_writer.writerow(header)

    norm_writer = None
    norm_f = None
    if normalize:
        norm_f = open(norm_path, "w", newline="", encoding="utf-8")
        norm_writer = csv.writer(norm_f)
        norm_writer.writerow(header)

    try:
        for result in extract_dataset_landmarks(image_root, limit_per_class=limit_per_class):
            stats["total_images"] += 1

            if result["status"] == "success":
                raw_writer.writerow(result["landmarks"] + [result["label"]])
                if normalize:
                    normalized_values = normalize_landmarks(result["landmarks"])
                    norm_writer.writerow(normalized_values + [result["label"]])
                stats["successful"] += 1
            elif result["status"] == "no_hand":
                stats["no_hand_detected"] += 1
                stats["no_hand_files"].append(result["image_path"])
            elif result["status"] == "unreadable":
                stats["unreadable"] += 1
                stats["unreadable_files"].append(result["image_path"])
    finally:
        raw_f.close()
        if norm_f:
            norm_f.close()

    stats["csv_file"] = str(output_csv_path)
    if normalize:
        stats["normalized_csv_file"] = str(norm_path)

    return stats