"""
preprocessing_service.py

Orchestrates the preprocessing pipeline: build the CSV, then generate
the validation report. This lives in app/services/ (not app/ai/)
because it doesn't touch cv2/mediapipe itself - it only calls into
the AI module's functions. This is what the /preprocess endpoint
calls, and what keeps preprocessing reusable for future datasets.
"""

from pathlib import Path
from typing import Optional

from app.ai.preprocessing.dataset_builder import build_dataset
from app.ai.preprocessing.validator import generate_report


class PreprocessingService:
    def __init__(
        self,
        image_root: Optional[Path] = None,
        csv_output_path: Optional[Path] = None,
        report_output_path: Optional[Path] = None,
    ):
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        self._image_root = image_root or (project_root / "datasets" / "asl_alphabet")
        self._csv_output_path = csv_output_path or (project_root / "backend" / "data" / "landmarks.csv")
        self._report_output_path = report_output_path or (project_root / "backend" / "data" / "dataset_report.json")

    def run(self, limit_per_class: Optional[int] = None, normalize: bool = False) -> dict:
        stats = build_dataset(
            self._image_root,
            self._csv_output_path,
            limit_per_class=limit_per_class,
            normalize=normalize,
        )
        report = generate_report(stats, self._report_output_path)

        result = {
            "images_processed": report["total_images_processed"],
            "successful": report["successful_detections"],
            "failed": report["no_hand_detected"] + report["unreadable_or_corrupted"],
            "csv_file": report["csv_file"],
        }
        if normalize:
            result["normalized_csv_file"] = stats.get("normalized_csv_file")

        return result