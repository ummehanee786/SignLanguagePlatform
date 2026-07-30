import pytest
from app.services.error_analysis_service import ErrorAnalysisService


class MockProgressService:
    def __init__(self, attempts):
        self._attempts = attempts

    def get_history(self, student_id, limit=None):
        # Return attempts sorted by timestamp descending to match real ProgressService
        sorted_attempts = sorted(self._attempts, key=lambda x: x["timestamp"], reverse=True)
        return sorted_attempts[:limit] if limit else sorted_attempts

    def get_dashboard(self, student_id):
        return {
            "weakest_alphabets": [
                {"alphabet": "X", "accuracy_percentage": 10.0, "attempts": 5}
            ]
        }


def test_error_analysis_metrics():
    # Setup test attempts chronologically
    attempts = [
        # Session 1
        {
            "student_id": "test_student",
            "session_id": "session-1",
            "alphabet_practiced": "M",
            "predicted_alphabet": "N",
            "correct": False,
            "confidence": 0.90,
            "timestamp": "2026-07-30T10:00:00Z",
        },
        {
            "student_id": "test_student",
            "session_id": "session-1",
            "alphabet_practiced": "M",
            "predicted_alphabet": "N",
            "correct": False,
            "confidence": 0.85,
            "timestamp": "2026-07-30T10:01:00Z",
        },
        {
            "student_id": "test_student",
            "session_id": "session-1",
            "alphabet_practiced": "A",
            "predicted_alphabet": "A",
            "correct": True,
            "confidence": 0.75,
            "timestamp": "2026-07-30T10:02:00Z",
        },
        {
            "student_id": "test_student",
            "session_id": "session-1",
            "alphabet_practiced": "A",
            "predicted_alphabet": "A",
            "correct": True,
            "confidence": 0.70,
            "timestamp": "2026-07-30T10:03:00Z",
        },
        # Session 2 - Repeated M mistake
        {
            "student_id": "test_student",
            "session_id": "session-2",
            "alphabet_practiced": "M",
            "predicted_alphabet": "N",
            "correct": False,
            "confidence": 0.80,
            "timestamp": "2026-07-30T11:00:00Z",
        },
        # low confidence for A (attempts 3 & 4)
        {
            "student_id": "test_student",
            "session_id": "session-2",
            "alphabet_practiced": "A",
            "predicted_alphabet": "A",
            "correct": True,
            "confidence": 0.65,
            "timestamp": "2026-07-30T11:01:00Z",
        },
        # performance trend for B (need 4 attempts)
        {
            "student_id": "test_student",
            "session_id": "session-2",
            "alphabet_practiced": "B",
            "predicted_alphabet": "X",
            "correct": False,
            "confidence": 0.90,
            "timestamp": "2026-07-30T11:02:00Z",
        },
        {
            "student_id": "test_student",
            "session_id": "session-2",
            "alphabet_practiced": "B",
            "predicted_alphabet": "X",
            "correct": False,
            "confidence": 0.90,
            "timestamp": "2026-07-30T11:03:00Z",
        },
        {
            "student_id": "test_student",
            "session_id": "session-2",
            "alphabet_practiced": "B",
            "predicted_alphabet": "B",
            "correct": True,
            "confidence": 0.90,
            "timestamp": "2026-07-30T11:04:00Z",
        },
        {
            "student_id": "test_student",
            "session_id": "session-2",
            "alphabet_practiced": "B",
            "predicted_alphabet": "B",
            "correct": True,
            "confidence": 0.90,
            "timestamp": "2026-07-30T11:05:00Z",
        },
    ]

    mock_progress_service = MockProgressService(attempts)
    service = ErrorAnalysisService(mock_progress_service)
    insight = service.analyze_student_errors("test_student")

    # Assertions
    assert insight.student_id == "test_student"

    # Confused pairs: M -> N should be top confused with count 3
    assert len(insight.most_confused_pairs) > 0
    assert insight.most_confused_pairs[0].expected == "M"
    assert insight.most_confused_pairs[0].predicted == "N"
    assert insight.most_confused_pairs[0].count == 3

    # Consistently low confidence: A has average confidence of (0.75 + 0.70 + 0.65) / 3 = 0.70 < 0.80
    assert len(insight.low_confidence_alphabets) == 1
    assert insight.low_confidence_alphabets[0].alphabet == "A"
    assert insight.low_confidence_alphabets[0].average_confidence == 0.70

    # Repeated mistakes: M is incorrect in session-1 and session-2
    assert len(insight.repeated_mistakes) == 1
    assert insight.repeated_mistakes[0].alphabet == "M"
    assert "session-1" in insight.repeated_mistakes[0].sessions
    assert "session-2" in insight.repeated_mistakes[0].sessions

    # Immediate revision:
    # M has 3 attempts, 0 correct -> accuracy 0% < 60% -> should be in the list
    assert "M" in insight.revision_required_gestures

    # Trends: B has 4 attempts: earlier 2: both incorrect (0%), later 2: both correct (100%)
    # Trend should be "improving"
    assert len(insight.performance_trends) == 1
    t = insight.performance_trends[0]
    assert t.alphabet == "B"
    assert t.trend == "improving"
    assert t.earlier_accuracy == 0.0
    assert t.later_accuracy == 100.0
