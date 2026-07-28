from typing import Optional, Dict

from pydantic import BaseModel

from app.schemas.lesson import LessonDetail


class SessionInfo(BaseModel):
    session_id: str
    student_id: str
    lesson_id: int
    current_lesson_id: int
    auto_next: bool
    attempts: int
    correct_attempts: int
    start_time: float
    end_time: Optional[float] = None


class StartPracticeResponse(BaseModel):
    """What POST /practice/start/{lesson_id} returns."""
    session: SessionInfo
    lesson: LessonDetail


class AssessmentRecordSchema(BaseModel):
    """
    The Assessment Result produced by the Sign Accuracy Assessment
    Engine (app/ai/assessment/sign_accuracy_engine.py) - a full record,
    not just a bare prediction.
    """
    expected_gesture: str
    predicted_gesture: Optional[str] = None
    correct: bool
    confidence: float
    gesture_accuracy: float  # student's historical accuracy on this letter
    attempt_number: int
    time_taken_seconds: float
    session_accuracy: float
    model_version: Optional[str] = None
    student_id: Optional[str] = None
    session_id: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    timestamp: str


class FeedbackObjectSchema(BaseModel):
    """The Feedback Object - human-facing message derived from the AssessmentRecord."""
    message: str
    tip: Optional[str] = None
    severity: str = "info"


class AttemptResponse(BaseModel):
    """
    What POST /practice/{session_id}/attempt returns.

    `attempt_recorded` distinguishes a real, graded attempt (assessment +
    feedback populated) from a failed capture - no hand detected, bad
    image - which never reaches the assessment engine at all (see
    assessment_service.submit_attempt).
    """
    session_id: str
    attempt_recorded: bool

    # Populated when attempt_recorded=False (a failed capture, not graded)
    attempt_number: Optional[int] = None
    session_accuracy: Optional[float] = None
    message: Optional[str] = None

    # Populated when attempt_recorded=True (a real, graded assessment)
    assessment: Optional[AssessmentRecordSchema] = None
    feedback: Optional[FeedbackObjectSchema] = None
    auto_advanced: bool = False
    next_lesson: Optional[LessonDetail] = None
    alphabet_complete: bool = False


class EndPracticeResponse(BaseModel):
    """What POST /practice/{session_id}/end returns."""
    session: SessionInfo
    duration_seconds: Optional[float] = None
    final_session_accuracy: float