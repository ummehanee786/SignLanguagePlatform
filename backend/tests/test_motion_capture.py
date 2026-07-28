"""
test_motion_capture.py

Unit tests for:
  - app/services/motion_capture_service.py (Task 1: continuous frame
    capture, rolling buffer, oldest-frame discard)
  - app/ai/assessment/motion_metrics.py (Task 2: motion-based
    assessment metrics)

Run with:
    cd d:\\SignLanguagePlatform\\backend
    ..\\venv\\Scripts\\python -m pytest tests/test_motion_capture.py -v
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import pytest

from app.ai.assessment.motion_metrics import (
    GestureFrameRecord,
    average_confidence,
    gesture_stability,
    invalid_frames_before_valid,
    overall_sign_score,
)
from app.services.motion_capture_service import MotionCaptureManager, MotionCaptureSession


# ---------------------------------------------------------------------------
# Test doubles - mirror the fields motion_capture_service.py actually reads
# off a PredictionResponse, without needing FastAPI/mediapipe installed.
# ---------------------------------------------------------------------------

@dataclass
class FakePrediction:
    success: bool
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    landmarks: Optional[List[float]] = None
    model_version: str = "test-v1"
    processing_time: float = 0.01
    probabilities: Optional[Dict[str, float]] = None
    error: Optional[str] = None


def _stable_landmarks(jitter: float = 0.0) -> list:
    """63 values with a non-degenerate wrist->middle-MCP distance (needed
    for normalize_landmarks' scale calculation to not divide by ~0)."""
    lm = [0.5] * 63
    lm[9 * 3: 9 * 3 + 3] = [0.5 + jitter, 0.6 + jitter, 0.0]
    return lm


# ---------------------------------------------------------------------------
# MotionCaptureSession (Task 1)
# ---------------------------------------------------------------------------

class TestMotionCaptureSessionBuffering:
    def test_discards_oldest_frame_once_full(self):
        session = MotionCaptureSession(max_frames=3)
        for i in range(5):
            session.add_prediction(
                FakePrediction(success=True, predicted_class="B", confidence=0.9,
                                landmarks=_stable_landmarks(jitter=i * 0.001))
            )
        assert len(session.sequence_buffer) == 3
        assert session.is_full()

    def test_invalid_frames_are_logged_but_not_added_to_sequence(self):
        session = MotionCaptureSession(max_frames=10)
        session.add_prediction(FakePrediction(success=False))
        session.add_prediction(FakePrediction(success=True, predicted_class="A",
                                                confidence=0.7, landmarks=_stable_landmarks()))
        assert len(session.records()) == 2          # both logged
        assert len(session.sequence_buffer) == 1     # only the valid one entered the sequence

    def test_clear_resets_both_buffers(self):
        session = MotionCaptureSession(max_frames=5)
        session.add_prediction(FakePrediction(success=True, predicted_class="A",
                                                confidence=0.9, landmarks=_stable_landmarks()))
        session.clear()
        assert len(session) == 0
        assert len(session.sequence_buffer) == 0

    def test_sequence_exposes_fixed_width_vectors(self):
        """Task 1: 'expose the complete sequence to any future temporal model'."""
        session = MotionCaptureSession(max_frames=5)
        session.add_prediction(FakePrediction(success=True, predicted_class="A",
                                                confidence=0.9, landmarks=_stable_landmarks()))
        sequence = session.sequence_buffer.frames()
        assert len(sequence) == 1
        assert sequence[0].shape == (63,)


class TestMotionCaptureManager:
    def test_get_or_create_returns_same_session(self):
        manager = MotionCaptureManager()
        a = manager.get_or_create("sess-1")
        b = manager.get_or_create("sess-1")
        assert a is b

    def test_reset_clears_without_dropping_the_session(self):
        manager = MotionCaptureManager()
        session = manager.get_or_create("sess-1")
        session.add_prediction(FakePrediction(success=True, predicted_class="A",
                                                confidence=0.9, landmarks=_stable_landmarks()))
        manager.reset("sess-1")
        assert len(manager.get("sess-1")) == 0
        assert manager.get("sess-1") is not None

    def test_remove_drops_the_session_entirely(self):
        manager = MotionCaptureManager()
        manager.get_or_create("sess-1")
        manager.remove("sess-1")
        assert manager.get("sess-1") is None


# ---------------------------------------------------------------------------
# motion_metrics.py (Task 2)
# ---------------------------------------------------------------------------

class TestInvalidFramesBeforeValid:
    def test_counts_only_leading_invalid_frames(self):
        records = [
            GestureFrameRecord(None, valid=False),
            GestureFrameRecord(None, valid=False),
            GestureFrameRecord(_stable_landmarks(), valid=True, confidence=0.9),
            GestureFrameRecord(None, valid=False),  # a later invalid frame - not counted
        ]
        assert invalid_frames_before_valid(records) == 2

    def test_zero_when_first_frame_is_already_valid(self):
        records = [GestureFrameRecord(_stable_landmarks(), valid=True, confidence=0.9)]
        assert invalid_frames_before_valid(records) == 0


class TestAverageConfidence:
    def test_averages_only_valid_frames(self):
        records = [
            GestureFrameRecord(_stable_landmarks(), valid=True, confidence=0.8),
            GestureFrameRecord(None, valid=False),
            GestureFrameRecord(_stable_landmarks(), valid=True, confidence=1.0),
        ]
        assert average_confidence(records) == 0.9

    def test_none_when_no_valid_frames(self):
        assert average_confidence([GestureFrameRecord(None, valid=False)]) is None


class TestGestureStability:
    def test_perfectly_still_hold_scores_near_100(self):
        records = [
            GestureFrameRecord(_stable_landmarks(), valid=True, confidence=0.9)
            for _ in range(5)
        ]
        score = gesture_stability(records)
        assert score > 99.0

    def test_unstable_hold_scores_much_lower(self):
        records = [
            GestureFrameRecord(_stable_landmarks(jitter=0.5 * i), valid=True, confidence=0.9)
            for i in range(5)
        ]
        score = gesture_stability(records)
        assert score < 40.0

    def test_none_with_fewer_than_two_valid_frames(self):
        records = [GestureFrameRecord(_stable_landmarks(), valid=True, confidence=0.9)]
        assert gesture_stability(records) is None


class TestOverallSignScore:
    def test_correct_confident_fast_stable_scores_high(self):
        score = overall_sign_score(
            handshape_correct=True, confidence=0.95, time_taken_seconds=1.5, stability=98.0,
        )
        assert score > 90.0

    def test_wrong_answer_scores_far_lower_than_correct_even_if_confident(self):
        wrong = overall_sign_score(
            handshape_correct=False, confidence=0.95, time_taken_seconds=1.5, stability=98.0,
        )
        correct = overall_sign_score(
            handshape_correct=True, confidence=0.95, time_taken_seconds=1.5, stability=98.0,
        )
        assert wrong < correct
        assert wrong < 60.0  # correctness dominates the weighting

    def test_works_without_stability_data(self):
        score = overall_sign_score(
            handshape_correct=True, confidence=0.9, time_taken_seconds=2.0, stability=None,
        )
        assert 0.0 <= score <= 100.0

    def test_score_is_bounded(self):
        score = overall_sign_score(
            handshape_correct=True, confidence=1.5, time_taken_seconds=0.0, stability=150.0,
        )
        assert 0.0 <= score <= 100.0
