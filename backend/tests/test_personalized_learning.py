from datetime import datetime, timezone, timedelta
import pytest
from app.services.progress_service import ProgressService
from app.services.recommendation_service import RecommendationService
from app.services.assessment_service import AssessmentService
from app.content.lesson_service import LessonService
from app.ai.assessment.sign_accuracy_engine import SignAccuracyAssessmentEngine
from app.schemas.prediction import PredictionResponse


class MockGestureService:
    def predict(self, image_bytes: bytes):
        # Always return successful prediction for class 'B'
        return PredictionResponse(
            success=True,
            predicted_class="B",
            confidence=0.95,
            processing_time=0.015,
            landmarks=[0.5] * 63,
            has_person=True,
            hand_count=1,
            upper_body_visible=True,
            partial_hand_visible=False,
            hand_centered=True,
        )


class MockSessionService:
    def __init__(self):
        self.session = {
            "session_id": "session-123",
            "student_id": "student-test",
            "lesson_id": 1,
            "current_lesson_id": 1,
            "current_lesson_started_at": 1000.0,
            "auto_next": True,
            "start_time": 1000.0,
            "end_time": None,
            "attempts": 0,
            "correct_attempts": 0,
        }

    def get_session(self, session_id):
        return self.session

    def record_attempt(self, session_id, correct):
        self.session["attempts"] += 1
        if correct:
            self.session["correct_attempts"] += 1
        return self.session

    def advance_lesson(self, session_id, next_lesson_id):
        self.session["current_lesson_id"] = next_lesson_id
        return self.session

    def time_since_lesson_shown(self, session):
        return 2.5

    def session_accuracy(self, session):
        if session["attempts"] == 0:
            return 0.0
        return round(100 * session["correct_attempts"] / session["attempts"], 2)


def test_learner_profile_bootstrapping_and_updates(tmp_path):
    # Setup paths
    attempts_file = tmp_path / "attempts.json"
    
    # Instantiate ProgressService with temp storage path
    progress = ProgressService(storage_path=attempts_file)
    student = "student-alice"

    # 1. Profile should start with 0 for new student
    profile = progress.get_learner_profile(student)
    assert profile["student_id"] == student
    assert profile["total_attempts"] == 0
    assert profile["total_practice_sessions"] == 0

    # 2. Record correct attempt on 'A'
    progress.record_attempt(
        student_id=student,
        alphabet_practiced="A",
        predicted_alphabet="A",
        correct=True,
        confidence=0.90,
        inference_time=0.01,
        session_id="sess-1"
    )

    profile = progress.get_learner_profile(student)
    assert profile["total_attempts"] == 1
    assert profile["total_practice_sessions"] == 1
    assert profile["consecutive_correct"]["A"] == 1
    assert profile["consecutive_incorrect"]["A"] == 0
    assert profile["alphabet_mastery"]["A"] == 1.0
    assert profile["average_confidence"]["A"] == 0.90

    # 3. Record incorrect attempt on 'A' (predicting 'B')
    progress.record_attempt(
        student_id=student,
        alphabet_practiced="A",
        predicted_alphabet="B",
        correct=False,
        confidence=0.80,
        inference_time=0.01,
        session_id="sess-1"
    )

    profile = progress.get_learner_profile(student)
    assert profile["total_attempts"] == 2
    assert profile["total_practice_sessions"] == 1
    assert profile["consecutive_correct"]["A"] == 0
    assert profile["consecutive_incorrect"]["A"] == 1
    assert profile["alphabet_mastery"]["A"] == 0.5   # 1 out of 2 correct
    assert profile["average_confidence"]["A"] == 0.85 # (0.90 + 0.80) / 2


def test_recommendation_rules(tmp_path):
    attempts_file = tmp_path / "attempts.json"
    progress = ProgressService(storage_path=attempts_file)
    student = "student-rec"

    # We need an ErrorAnalysisService and RecommendationService
    from app.services.error_analysis_service import ErrorAnalysisService
    ea_service = ErrorAnalysisService(progress)
    rec_service = RecommendationService(progress, ea_service)

    # Let's populate the history to trigger different recommendations:
    # A: practiced recently, but low confidence (5 correct attempts, conf=0.75)
    for _ in range(3):
        progress.record_attempt(
            student_id=student,
            alphabet_practiced="A",
            predicted_alphabet="A",
            correct=True,
            confidence=0.75,
            inference_time=0.01
        )
    
    # B: experienced frequent confusion with C
    progress.record_attempt(
        student_id=student,
        alphabet_practiced="B",
        predicted_alphabet="C",
        correct=False,
        confidence=0.85,
        inference_time=0.01
    )
    progress.record_attempt(
        student_id=student,
        alphabet_practiced="B",
        predicted_alphabet="C",
        correct=False,
        confidence=0.85,
        inference_time=0.01
    )

    # D: low mastery (1 correct out of 4 attempts)
    progress.record_attempt(
        student_id=student,
        alphabet_practiced="D",
        predicted_alphabet="X",
        correct=False,
        confidence=0.80,
        inference_time=0.01
    )
    progress.record_attempt(
        student_id=student,
        alphabet_practiced="D",
        predicted_alphabet="D",
        correct=True,
        confidence=0.85,
        inference_time=0.01
    )
    progress.record_attempt(
        student_id=student,
        alphabet_practiced="D",
        predicted_alphabet="Y",
        correct=False,
        confidence=0.80,
        inference_time=0.01
    )

    # Run get_recommendations
    recs = rec_service.get_recommendations(student)

    # D should yield "Low mastery level" (mastery = 33%)
    d_recs = [r for r in recs if r.alphabet == "D"]
    assert len(d_recs) > 0
    assert d_recs[0].reason == "Low mastery level"

    # B should yield "Frequent confusion with 'C'"
    b_recs = [r for r in recs if r.alphabet == "B"]
    assert len(b_recs) > 0
    assert b_recs[0].reason == "Frequent confusion with 'C'"

    # A should yield "Low confidence despite correct predictions" (consec_correct = 3, avg confidence = 0.75)
    a_recs = [r for r in recs if r.alphabet == "A"]
    if a_recs:
        assert a_recs[0].reason == "Low confidence despite correct predictions"


def test_dynamic_lesson_progression(tmp_path):
    attempts_file = tmp_path / "attempts.json"
    progress = ProgressService(storage_path=attempts_file)
    
    from app.services.error_analysis_service import ErrorAnalysisService
    ea_service = ErrorAnalysisService(progress)
    rec_service = RecommendationService(progress, ea_service)

    # Setup assessment engine and service
    engine = SignAccuracyAssessmentEngine(progress)
    sess_service = MockSessionService()
    lesson_service = LessonService()
    gesture_service = MockGestureService()

    # Pre-populate recommendations for 'student-test':
    # Let's say they have low mastery on 'C'
    progress.record_attempt(
        student_id="student-test",
        alphabet_practiced="C",
        predicted_alphabet="X",
        correct=False,
        confidence=0.90,
        inference_time=0.02
    )

    # The student is currently practicing 'A' (Lesson 1) in the mock session.
    # We submit a correct attempt for 'B' (which is expected because MockGestureService predicts 'B').
    # But wait, the session expects current_lesson_id = 1 (meaning expected is 'A').
    # Let's mock a correct attempt where expected matches predicted ('B').
    # We set session current_lesson_id = 2 ('B')
    sess_service.session["current_lesson_id"] = 2 # expected: 'B'

    # Instantiate AssessmentService
    service = AssessmentService(
        lesson_service=lesson_service,
        session_service=sess_service,
        gesture_service=gesture_service,
        assessment_engine=engine
    )

    # Mock the recommendation service in the assessment flow
    # Since C is low mastery, it should be top recommended
    # Let's trigger the attempt submit
    result = service.submit_attempt(
        session_id="session-123",
        image_bytes=b"dummy_bytes"
    )

    assert result["attempt_recorded"] is True
    assert result["assessment"]["correct"] is True
    # The session should have advanced to 'C' (Lesson ID 3) because it was recommended!
    assert result["auto_advanced"] is True
    assert result["next_lesson"]["sign"] == "C"
