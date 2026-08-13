"""
assessment_service.py

Manages the practice/assessment workflow sequence:

    User selects lesson -> Lesson displayed -> User clicks Practice ->
    Camera opens -> Landmarks extracted -> (placeholder response) ->
    Session ends

This service doesn't do any recognition itself - it orchestrates
SessionService (tracking the attempt) and GestureService (currently a
placeholder prediction). When the real classifier is ready, only
GestureService changes internally - this workflow stays the same.
"""

from typing import Optional

from app.content.lesson_service import LessonService
from app.services.session_service import SessionService
from app.services.gesture_service import GestureService


class AssessmentService:
    def __init__(
        self,
        lesson_service: LessonService,
        session_service: SessionService,
        gesture_service: GestureService,
    ):
        self._lesson_service = lesson_service
        self._session_service = session_service
        self._gesture_service = gesture_service

    def start_practice(self, lesson_id: int) -> Optional[dict]:
        """User selects lesson -> clicks Practice -> session starts."""
        lesson = self._lesson_service.get_lesson_by_id(lesson_id)
        if lesson is None:
            return None

        session = self._session_service.start_session(lesson_id)
        return {"session": session, "lesson": lesson}

    def submit_attempt(self, session_id: str):
        """Camera opens -> landmarks extracted -> placeholder response returned."""
        session = self._session_service.record_attempt(session_id)
        if session is None:
            return None

        prediction = self._gesture_service.predict()
        return {"session": session, "prediction": prediction}

    def end_practice(self, session_id: str) -> Optional[dict]:
        """Session ends."""
        return self._session_service.end_session(session_id)