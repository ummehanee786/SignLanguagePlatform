"""
assessment_service.py

Manages the real-time practice/assessment workflow:

    User selects alphabet -> Reference sign displayed -> User performs sign
    -> Webcam captures frame -> Model predicts gesture -> Gesture Comparison
    -> Accuracy Score Calculation -> Assessment Result -> Feedback Object ->
    Store attempt -> Move to next letter

Orchestrates four collaborators, each with one job:
    - LessonService: what letter is being shown / what's next (content)
    - SessionService: this session's attempt count, running accuracy, and
                       the "how long has this letter been shown" clock
    - GestureService: Live Prediction (AI, unchanged since Sprint 1)
    - SignAccuracyAssessmentEngine: Gesture Comparison, Accuracy Score
                       Calculation, Assessment Result, Feedback Object,
                       and storing the result as an Assessment Record
                       (app/ai/assessment/sign_accuracy_engine.py)

This file itself doesn't do any comparison or scoring anymore - that
logic now lives in the assessment engine, matching the SRS's own
separation of "Live Prediction" from "Gesture Comparison" onward.
"""

from typing import Optional

from app.content.lesson_service import LessonService
from app.services.session_service import SessionService
from app.services.gesture_service import GestureService
from app.services.motion_capture_service import MotionCaptureManager, get_motion_capture_manager
from app.ai.assessment.sign_accuracy_engine import SignAccuracyAssessmentEngine


class AssessmentService:
    def __init__(
        self,
        lesson_service: LessonService,
        session_service: SessionService,
        gesture_service: GestureService,
        assessment_engine: SignAccuracyAssessmentEngine,
        motion_capture_manager: Optional[MotionCaptureManager] = None,
    ):
        self._lesson_service = lesson_service
        self._session_service = session_service
        self._gesture_service = gesture_service
        self._assessment_engine = assessment_engine
        # Task 1/2: the rolling per-session frame window built from
        # POST /practice/{session_id}/stream-frame calls (if the caller
        # made any) - defaults to the shared singleton so this service
        # sees the SAME buffers that endpoint fills, without every
        # caller needing to wire it through by hand.
        self._motion_capture_manager = motion_capture_manager or get_motion_capture_manager()

    def start_practice(
        self, lesson_id: int, student_id: str, auto_next: bool = True
    ) -> Optional[dict]:
        """User selects an alphabet -> reference sign is displayed -> session starts."""
        lesson = self._lesson_service.get_lesson_by_id(lesson_id)
        if lesson is None:
            return None

        session = self._session_service.start_session(lesson_id, student_id, auto_next)
        return {"session": session, "lesson": lesson}

    def submit_attempt(self, session_id: str, image_bytes: bytes) -> Optional[dict]:
        """
        Webcam captures a frame -> Live Prediction -> (if successful)
        Gesture Comparison + Accuracy Scoring via the assessment engine
        -> store Assessment Record -> (maybe) move to next letter.

        If the model couldn't make a prediction at all (no hand
        detected, corrupt image), this never reaches the assessment
        engine - it's not a graded attempt, the attempt counter and
        session accuracy are untouched, and the caller gets a message
        telling them to try again.
        """
        session = self._session_service.get_session(session_id)
        if session is None:
            return None

        current_lesson = self._lesson_service.get_lesson_by_id(session["current_lesson_id"])
        expected_alphabet = current_lesson["sign"]

        # Live Prediction
        prediction = self._gesture_service.predict(image_bytes)

        if not prediction.success:
            return {
                "session_id": session_id,
                "attempt_recorded": False,
                "attempt_number": session["attempts"],
                "session_accuracy": self._session_service.session_accuracy(session),
                "message": prediction.error or "Could not read a gesture - please try again.",
            }

        time_taken_seconds = self._session_service.time_since_lesson_shown(session)
        attempt_number = session["attempts"] + 1  # this attempt, about to be recorded

        # Task 1/2: whatever was streamed via /stream-frame while the
        # student held this sign (may be None/empty if they never
        # streamed anything - assess() degrades gracefully either way).
        capture = self._motion_capture_manager.get(session_id)
        motion_records = capture.records() if capture is not None else None

        # Gesture Comparison -> Accuracy Score Calculation -> Assessment
        # Result -> Feedback Object -> stored as an Assessment Record
        record, feedback = self._assessment_engine.assess(
            student_id=session["student_id"],
            session_id=session_id,
            expected_gesture=expected_alphabet,
            prediction=prediction,
            attempt_number=attempt_number,
            time_taken_seconds=time_taken_seconds,
            # session_accuracy computed AFTER this attempt is folded in below,
            # so pass a placeholder for now and overwrite the record's copy.
            session_accuracy=0.0,
            motion_records=motion_records,
        )

        # This attempt has been graded - reset the capture window so the
        # NEXT gesture (whether that's a retry of this letter or the
        # next one) starts with a clean window instead of blending in
        # frames from the sign that was just judged.
        self._motion_capture_manager.reset(session_id)

        session = self._session_service.record_attempt(session_id, correct=record.correct)
        record.session_accuracy = self._session_service.session_accuracy(session)

        next_lesson_id = None
        if session["auto_next"] and record.correct:
            next_lesson_id = self._lesson_service.get_next_lesson_id(session["current_lesson_id"])
            session = self._session_service.advance_lesson(session_id, next_lesson_id)
        else:
            # Incorrect (or auto_next disabled): stay on the same letter,
            # but restart the "time taken" clock so the next retry gets
            # its own fresh measurement instead of accumulating time
            # across multiple attempts at the same letter.
            self._session_service.reset_lesson_timer(session_id)

        next_lesson = (
            self._lesson_service.get_lesson_by_id(next_lesson_id) if next_lesson_id else None
        )
        alphabet_complete = bool(session["auto_next"] and record.correct and next_lesson_id is None)

        return {
            "session_id": session_id,
            "attempt_recorded": True,
            "assessment": record.to_dict(),
            "feedback": feedback.to_dict(),
            "auto_advanced": next_lesson_id is not None,
            "next_lesson": next_lesson,
            "alphabet_complete": alphabet_complete,
        }

    def end_practice(self, session_id: str) -> Optional[dict]:
        """Session ends - returns a final summary."""
        session = self._session_service.end_session(session_id)
        if session is None:
            return None

        # No more attempts coming for this session - drop its capture
        # window entirely rather than just clearing it, so it doesn't
        # sit in memory indefinitely for a session that's now over.
        self._motion_capture_manager.remove(session_id)

        duration = None
        if session["end_time"] is not None and session["start_time"] is not None:
            duration = round(session["end_time"] - session["start_time"], 2)

        return {
            "session": session,
            "duration_seconds": duration,
            "final_session_accuracy": self._session_service.session_accuracy(session),
        }