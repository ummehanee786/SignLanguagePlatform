"""
motion_metrics.py

Task 2: motion-based assessment metrics, inspired by the SRS. These
operate on a short WINDOW of per-frame captures gathered while the
student holds a sign (see app/services/motion_capture_service.py) -
distinct from the single graded frame submit_attempt() classifies.

Deliberately pure functions + a plain dataclass (no side effects, no
FastAPI/pydantic dependency) so each metric is independently unit
testable with synthetic frame records - same reasoning as
app/ai/feedback/models.py and PredictionResult.

Stability is computed over the NORMALIZED landmark representation
(app.ai.preprocessing.extract_landmarks.normalize_landmarks) rather
than raw coordinates - the same wrist-centered, scale-invariant space
the classifier itself sees - so "fluctuation" reflects actual handshape
instability, not the hand drifting toward/away from the camera.
"""

import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GestureFrameRecord:
    """
    One streamed frame's outcome, as captured by MotionCaptureSession
    while the student holds a sign.

    normalized_landmarks : the 63-value normalized vector, or None if
                            this frame was invalid (no hand / bad landmarks).
    valid                : whether a prediction was actually made for
                            this frame (mirrors PredictionResponse.success).
    predicted_class      : this frame's predicted letter, if valid.
    confidence           : this frame's model confidence, if valid.
    """
    normalized_landmarks: Optional[List[float]]
    valid: bool
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None


def invalid_frames_before_valid(records: List[GestureFrameRecord]) -> int:
    """
    Counts leading invalid frames - how many frames failed (no hand
    detected, bad landmarks) before the FIRST usable one - not the
    total invalid-frame count across the whole window. This measures
    "how long did it take the student to get their hand in frame",
    which is what the SRS metric is actually about; invalid frames
    scattered elsewhere in the sequence (e.g. a brief hand wobble out of
    frame mid-hold) are a separate stability concern, not this one.
    """
    count = 0
    for record in records:
        if record.valid:
            break
        count += 1
    return count


def average_confidence(records: List[GestureFrameRecord]) -> Optional[float]:
    """Mean model confidence across every VALID frame in the window, or None if none were valid."""
    confidences = [r.confidence for r in records if r.valid and r.confidence is not None]
    if not confidences:
        return None
    return round(sum(confidences) / len(confidences), 4)


# Tuned so a natural, steady hold (small sub-threshold jitter, mean
# frame-to-frame displacement roughly 0.01-0.05 in normalized-landmark
# space) scores in the 70-95 range, while a genuinely unstable hold
# (the hand visibly moving or reshaping, displacement > ~0.3) drops
# below 40 - see tests/test_motion_metrics.py for the concrete cases
# this was checked against.
_STABILITY_DECAY_RATE = 8.0


def gesture_stability(records: List[GestureFrameRecord]) -> Optional[float]:
    """
    0-100 score, higher = more stable (less handshape fluctuation) while
    holding the sign. None if fewer than 2 valid frames were captured -
    fluctuation isn't a meaningful concept from a single sample.

    Computed as the mean Euclidean distance between consecutive valid
    frames' normalized landmark vectors, mapped onto a bounded 0-100
    scale via exponential decay (rather than a linear one) so the score
    degrades smoothly and never goes negative, regardless of how large
    the raw displacement gets.
    """
    valid_vectors = [
        r.normalized_landmarks for r in records
        if r.valid and r.normalized_landmarks is not None
    ]
    if len(valid_vectors) < 2:
        return None

    displacements = [
        math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
        for v1, v2 in zip(valid_vectors, valid_vectors[1:])
    ]
    mean_displacement = sum(displacements) / len(displacements)

    score = 100.0 * math.exp(-_STABILITY_DECAY_RATE * mean_displacement)
    return round(score, 2)


def overall_sign_score(
    handshape_correct: bool,
    confidence: float,
    time_taken_seconds: float,
    stability: Optional[float] = None,
    target_time_seconds: float = 3.0,
) -> float:
    """
    Combines hand-shape accuracy, model confidence, and timing (plus
    stability, when a motion window is available) into ONE 0-100
    assessment score - the SRS's "overall sign score".

    Correctness dominates the weighting on purpose: a fast, stable,
    confident WRONG sign should still score far below a correct one.

        50% correctness (55% if no stability data)
        25% confidence  (28% if no stability data)
        15% timing      (17% if no stability data)
        10% stability (only when available)

    Timing scores 100 at or under `target_time_seconds` and decays
    smoothly (not a hard cliff) for slower attempts, since being a
    little slow is a much smaller problem than being wrong.
    """
    correctness_component = 100.0 if handshape_correct else 0.0
    confidence_component = max(0.0, min(1.0, confidence)) * 100.0

    if time_taken_seconds <= target_time_seconds:
        timing_component = 100.0
    else:
        overrun_ratio = (time_taken_seconds - target_time_seconds) / target_time_seconds
        timing_component = max(0.0, 100.0 * math.exp(-0.5 * overrun_ratio))

    if stability is not None:
        score = (
            0.50 * correctness_component
            + 0.25 * confidence_component
            + 0.15 * timing_component
            + 0.10 * stability
        )
    else:
        score = (
            0.55 * correctness_component
            + 0.28 * confidence_component
            + 0.17 * timing_component
        )

    return round(max(0.0, min(100.0, score)), 2)
