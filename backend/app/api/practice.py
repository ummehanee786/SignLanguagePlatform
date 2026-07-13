from fastapi import APIRouter, HTTPException

from app.content.lesson_service import LessonService
from app.services.session_service import SessionService
from app.services.gesture_service import GestureService
from app.services.assessment_service import AssessmentService

router = APIRouter()

_lesson_service = LessonService()
_session_service = SessionService()
_gesture_service = GestureService()
assessment_service = AssessmentService(_lesson_service, _session_service, _gesture_service)


@router.post("/practice/start/{lesson_id}")
def start_practice(lesson_id: int):
    """User selects a lesson and clicks Practice - starts a session."""
    result = assessment_service.start_practice(lesson_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Lesson {lesson_id} not found")
    return result


@router.post("/practice/{session_id}/attempt")
def submit_attempt(session_id: str):
    """Camera captures a frame -> landmarks extracted -> placeholder prediction returned."""
    result = assessment_service.submit_attempt(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return result


@router.post("/practice/{session_id}/end")
def end_practice(session_id: str):
    """Ends the practice session."""
    result = assessment_service.end_practice(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return result