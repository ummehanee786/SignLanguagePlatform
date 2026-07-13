"""
validator.py

Task 3: takes the stats collected during dataset_builder's single pass
over the images and turns them into a data quality report, saved as
dataset_report.json. Kept separate from dataset_builder because
"build the data" and "report on data quality" are different
responsibilities, even though they share the same underlying numbers.
"""

import json
from pathlib import Path


def generate_report(stats: dict, output_path: Path) -> dict:
    total = stats["total_images"]
    successful = stats["successful"]

    success_percentage = round((successful / total) * 100, 2) if total > 0 else 0.0

    report = {
        "total_images_processed": total,
        "successful_detections": successful,
        "no_hand_detected": stats["no_hand_detected"],
        "unreadable_or_corrupted": stats["unreadable"],
        "success_percentage": success_percentage,
        "csv_file": stats.get("csv_file"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report