"""
feature_pipeline.py

"Feature Validation" and "Feature Normalization" steps of the inference
pipeline:

    21 Hand Landmarks -> Feature Validation -> Feature Normalization
    -> 63-Dimensional Feature Vector

Normalization math is NOT reimplemented here - it's imported from
app.ai.preprocessing.extract_landmarks.normalize_landmarks, the exact
function used to build the training dataset. If this module had its own
copy of that math, a future change to one copy and not the other would
silently create a train/serve skew bug (the model would be evaluated on
features computed differently than what it was trained on) - one of the
most common and hardest-to-notice bugs in production ML systems.
"""

from typing import List

import pandas as pd

from app.ai.preprocessing.extract_landmarks import normalize_landmarks
from app.ai.ml.training.data_utils import get_feature_columns

EXPECTED_LANDMARK_COUNT = 21
EXPECTED_VALUE_COUNT = EXPECTED_LANDMARK_COUNT * 3  # x, y, z per landmark

# MediaPipe's x/y are image-relative (~0-1) and z is a rough relative
# depth: legitimate values rarely leave roughly [-2, 2]. This is a sanity
# check to catch corrupted/garbage data (e.g. a caller passing pixel
# coordinates instead of MediaPipe's normalized ones), not a strict
# physical bound.
PLAUSIBLE_VALUE_RANGE = (-2.0, 2.0)


class FeatureValidationError(ValueError):
    """
    Raised when raw landmarks fail validation before normalization -
    e.g. wrong length, non-numeric values, or values outside a
    plausible range. Callers (engine.py) catch this and turn it into a
    graceful PredictionResult rather than letting it crash the request.
    """


def validate_landmarks(flat_values: List[float]) -> None:
    """
    Sanity-checks raw (unnormalized) landmark values before they're
    normalized and fed to the model. Raises FeatureValidationError on
    any problem; returns None (i.e. "fine, continue") otherwise.
    """
    if flat_values is None:
        raise FeatureValidationError("No landmarks provided (no hand detected in image).")

    if len(flat_values) != EXPECTED_VALUE_COUNT:
        raise FeatureValidationError(
            f"Expected {EXPECTED_VALUE_COUNT} landmark values "
            f"({EXPECTED_LANDMARK_COUNT} landmarks x 3 coords), got {len(flat_values)}."
        )

    low, high = PLAUSIBLE_VALUE_RANGE
    for i, value in enumerate(flat_values):
        if not isinstance(value, (int, float)):
            raise FeatureValidationError(f"Landmark value at index {i} is not numeric: {value!r}")
        if value != value:  # NaN check (NaN is the only value that != itself)
            raise FeatureValidationError(f"Landmark value at index {i} is NaN.")
        if value in (float("inf"), float("-inf")):
            raise FeatureValidationError(f"Landmark value at index {i} is infinite.")
        if not (low <= value <= high):
            raise FeatureValidationError(
                f"Landmark value at index {i} ({value}) is outside the plausible "
                f"range {PLAUSIBLE_VALUE_RANGE} - possible garbage input or a "
                "detector/coordinate-system mismatch."
            )


def build_feature_vector(flat_values: List[float]) -> pd.DataFrame:
    """
    Full Feature Validation + Feature Normalization step. Takes the raw
    63 values straight out of HandLandmarkDetector.extract_landmarks()
    and returns a single-row DataFrame with the exact column names/order
    the model was trained on (x0, y0, z0, ..., x20, y20, z20), ready to
    pass directly to model.predict() / model.predict_proba().

    Raises FeatureValidationError if the input doesn't pass validation.
    """
    validate_landmarks(flat_values)
    normalized = normalize_landmarks(flat_values)
    return pd.DataFrame([normalized], columns=get_feature_columns())