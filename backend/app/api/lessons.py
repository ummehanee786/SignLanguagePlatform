from fastapi import APIRouter, HTTPException

from app.content.lesson_service import LessonService
from app.schemas.lesson import LessonSummary, LessonDetail

router = APIRouter()

lesson_service = LessonService()


@router.get("/lessons", response_model=list[LessonSummary])
def get_lessons():
    """Returns all available lessons (summary view: id + sign only)."""
    return lesson_service.get_all_lessons()


@router.get("/lessons/{lesson_id}", response_model=LessonDetail)
def get_lesson(lesson_id: int):
    """Returns full details of one lesson."""
    lesson = lesson_service.get_lesson_by_id(lesson_id)
    if lesson is None:
        raise HTTPException(status_code=404, detail=f"Lesson {lesson_id} not found")
    return lesson