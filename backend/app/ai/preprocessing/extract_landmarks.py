"""
extract_landmarks.py (AI module version)

Task 1: pure landmark extraction - traverses every class folder,
reads every image, detects the hand, extracts and flattens the 21
landmarks into 63 values, and pairs each with its class label.

Deliberately does NOT save anything to disk - that's dataset_builder's
job. This module's only responsibility is producing the data.

This lives inside app/ai/ (not app/services/) because it uses
HandLandmarkDetector, which uses MediaPipe/OpenCV - per the "AI code
stays inside the AI module" rule.
"""

from pathlib import Path
from typing import Iterator, Optional

import cv2

from app.ai.handtracking.detector import HandLandmarkDetector


def normalize_landmarks(flat_values: list) -> list:
    """
    Makes landmarks less sensitive to the hand's absolute position/size
    in the frame:
      1. Translation invariance: shift so the wrist (landmark 0)
         becomes the origin (0, 0, 0).
      2. Scale invariance: divide by the wrist-to-middle-MCP (landmark 9)
         distance, a stable "hand size" reference.
    """
    points = [flat_values[i:i + 3] for i in range(0, len(flat_values), 3)]
    wrist = points[0]

    translated = [
        [p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]]
        for p in points
    ]

    scale = (
        translated[9][0] ** 2 + translated[9][1] ** 2 + translated[9][2] ** 2
    ) ** 0.5
    if scale < 1e-6:
        scale = 1e-6

    normalized = [[p[0] / scale, p[1] / scale, p[2] / scale] for p in translated]

    flat_normalized = []
    for p in normalized:
        flat_normalized.extend(p)
    return flat_normalized


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


def find_first_existing(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def extract_dataset_landmarks(
    image_root: Path,
    limit_per_class: Optional[int] = None,
    min_detection_confidence: float = 0.3,
) -> Iterator[dict]:
    """
    ...
    """
    # Prefer the asl_alphabet_train subfolder if it exists (the ASL
    # Alphabet dataset splits into train/test folders) - otherwise
    # fall back to treating image_root itself as the folder to scan.
    image_root = find_first_existing([
        image_root / "asl_alphabet_train",
        image_root,
    ])
    image_root = resolve_class_root(image_root)
    class_dirs = sorted([d for d in image_root.iterdir() if d.is_dir()])
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    detector = HandLandmarkDetector(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
    )

    try:
        for class_dir in class_dirs:
            label = class_dir.name
            image_files = [
                f for f in sorted(class_dir.iterdir())
                if f.is_file() and f.suffix.lower() in valid_extensions
            ]
            if limit_per_class:
                image_files = image_files[:limit_per_class]

            for image_path in image_files:
                image = cv2.imread(str(image_path))

                if image is None:
                    yield {
                        "image_path": str(image_path),
                        "label": label,
                        "landmarks": None,
                        "status": "unreadable",
                    }
                    continue

                landmarks = detector.extract_landmarks(image)

                if landmarks is None:
                    yield {
                        "image_path": str(image_path),
                        "label": label,
                        "landmarks": None,
                        "status": "no_hand",
                    }
                    continue

                yield {
                    "image_path": str(image_path),
                    "label": label,
                    "landmarks": landmarks,
                    "status": "success",
                }
    finally:
        detector.close()