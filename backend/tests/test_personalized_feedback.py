import pytest
from app.ai.assessment.sign_accuracy_engine import SignAccuracyAssessmentEngine
from app.schemas.prediction import PredictionResponse
from app.services.progress_service import ProgressService


class MockProgressServiceForTutor:
    def __init__(self, history_attempts, dashboard_weakest=None):
        self._attempts = history_attempts
        self._dashboard_weakest = dashboard_weakest or []

    def get_history(self, student_id, limit=None):
        return sorted(self._attempts, key=lambda x: x["timestamp"], reverse=True)

    def record_attempt(self, **kwargs):
        # Mock registration of attempt, do nothing
        pass

    def get_gesture_accuracy(self, student_id, alphabet):
        return 50.0

    def get_dashboard(self, student_id):
        return {
            "weakest_alphabets": [
                {"alphabet": w, "accuracy_percentage": 30.0, "attempts": 5}
                for w in self._dashboard_weakest
            ]
        }


def test_tutor_feedback_frequent_confusion():
    # Setup history: student has M -> N confusion before
    history = [
        {
            "student_id": "student-1",
            "session_id": "session-1",
            "alphabet_practiced": "M",
            "predicted_alphabet": "N",
            "correct": False,
            "confidence": 0.85,
            "timestamp": "2026-07-30T10:00:00Z",
        },
        {
            "student_id": "student-1",
            "session_id": "session-1",
            "alphabet_practiced": "M",
            "predicted_alphabet": "N",
            "correct": False,
            "confidence": 0.85,
            "timestamp": "2026-07-30T10:01:00Z",
        },
    ]

    mock_progress = MockProgressServiceForTutor(history)
    engine = SignAccuracyAssessmentEngine(mock_progress)

    # Let's perform an attempt with same confusion
    pred = PredictionResponse(
        success=True,
        predicted_class="N",
        confidence=0.88,
        processing_time=0.01,
        landmarks=[0.5] * 63,
        has_person=True,
        hand_count=1,
        upper_body_visible=True,
        partial_hand_visible=False,
        hand_centered=True,
    )

    record, feedback = engine.assess(
        student_id="student-1",
        session_id="session-1",
        expected_gesture="M",
        prediction=pred,
        attempt_number=3,
        time_taken_seconds=2.5,
        session_accuracy=0.0
    )

    # Message should contain the tutor note about frequent confusion
    assert "frequently confuse 'M' with 'N'" in feedback.message
    # And have recommendations
    assert len(feedback.recommendations) > 0
    assert "Revise 'M'" in feedback.recommendations[0]


def test_tutor_feedback_trends_and_recommendations():
    # Setup history: B has 4 attempts, improving
    history = [
        # Chronological order of timestamps will make earlier 2 attempts incorrect and later 2 attempts correct
        {
            "student_id": "student-1",
            "session_id": "session-1",
            "alphabet_practiced": "B",
            "predicted_alphabet": "X",
            "correct": False,
            "confidence": 0.80,
            "timestamp": "2026-07-30T10:00:00Z",
        },
        {
            "student_id": "student-1",
            "session_id": "session-1",
            "alphabet_practiced": "B",
            "predicted_alphabet": "X",
            "correct": False,
            "confidence": 0.80,
            "timestamp": "2026-07-30T10:01:00Z",
        },
        {
            "student_id": "student-1",
            "session_id": "session-1",
            "alphabet_practiced": "B",
            "predicted_alphabet": "B",
            "correct": True,
            "confidence": 0.85,
            "timestamp": "2026-07-30T10:02:00Z",
        },
        {
            "student_id": "student-1",
            "session_id": "session-1",
            "alphabet_practiced": "B",
            "predicted_alphabet": "B",
            "correct": True,
            "confidence": 0.85,
            "timestamp": "2026-07-30T10:03:00Z",
        },
    ]

    mock_progress = MockProgressServiceForTutor(history, dashboard_weakest=["K", "Y"])
    engine = SignAccuracyAssessmentEngine(mock_progress)

    pred = PredictionResponse(
        success=True,
        predicted_class="B",
        confidence=0.90,
        processing_time=0.01,
        landmarks=[0.5] * 63,
        has_person=True,
        hand_count=1,
        upper_body_visible=True,
        partial_hand_visible=False,
        hand_centered=True,
    )

    record, feedback = engine.assess(
        student_id="student-1",
        session_id="session-1",
        expected_gesture="B",
        prediction=pred,
        attempt_number=5,
        time_taken_seconds=1.2,
        session_accuracy=50.0
    )

    # Message should report improvement trend
    assert "improving on 'B'" in feedback.message
    # Recommendations should contain dashboard weakest recommendations
    assert any("Practice 'K'" in r for r in feedback.recommendations)
    assert any("Practice 'Y'" in r for r in feedback.recommendations)
