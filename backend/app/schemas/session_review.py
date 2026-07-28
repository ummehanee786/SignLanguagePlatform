"""
session_review.py  (app/schemas/)

Pydantic response schema for GET /practice/{session_id}/review.
"""

from typing import List, Optional

from pydantic import BaseModel


class ConfidencePoint(BaseModel):
    """One data point in the confidence trend chart."""
    attempt_number: int
    confidence: float
    correct: bool
    expected_gesture: str


class MistakeEntry(BaseModel):
    """A repeated mistake pattern observed during the session."""
    expected: str
    predicted: str
    count: int


class GestureFeedbackEntry(BaseModel):
    """Last recorded feedback for one gesture practiced in the session."""
    gesture: str
    attempts: int
    correct: int
    accuracy_percentage: float
    last_feedback_message: Optional[str] = None
    last_feedback_tip: Optional[str] = None


class SessionReviewResponse(BaseModel):
    """
    What GET /practice/{session_id}/review returns.

    Gives a full post-session breakdown so the learner (and any
    frontend review screen) can understand what happened, where
    errors occurred, and what to practise next.
    """
    session_id: str
    student_id: str

    # --- Summary ---
    total_attempts: int
    correct_attempts: int
    incorrect_attempts: int
    overall_score: float                        # percentage

    # --- Gesture lists ---
    correct_gestures: List[str]                 # distinct letters answered correctly
    incorrect_gestures: List[str]               # distinct letters that had at least one error

    # --- Confidence trend (ordered by attempt_number) ---
    confidence_trend: List[ConfidencePoint]

    # --- Error analysis ---
    most_common_mistakes: List[MistakeEntry]    # top 5 by count

    # --- Gesture-specific feedback ---
    gesture_feedback: List[GestureFeedbackEntry]

    # --- Next-steps recommendations ---
    recommended_gestures: List[str]             # up to 5 weakest gestures for next session
