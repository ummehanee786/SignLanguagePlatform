from fastapi import APIRouter, Query

from app.services.progress_service import get_progress_service
from app.services.error_analysis_service import get_error_analysis_service
from app.schemas.progress import DashboardResponse, AttemptRecord
from app.schemas.error_analysis import ErrorAnalysisInsight

router = APIRouter()

_progress_service = get_progress_service()
_error_analysis_service = get_error_analysis_service()


@router.get("/progress/{student_id}/dashboard", response_model=DashboardResponse)
def get_dashboard(student_id: str):
    """
    Task 2 deliverable: total attempts, accuracy %, most-mistaken
    alphabets, strongest/weakest alphabets, daily practice streak,
    average confidence, and recent practice history for one student.
    """
    return _progress_service.get_dashboard(student_id)


@router.get("/progress/{student_id}/error-analysis", response_model=ErrorAnalysisInsight)
def get_error_analysis(student_id: str):
    """
    Task 1: returns structured JSON error analysis and insights
    for the specified student, including most confused pairs, lowest
    confidence alphabets, repeated mistakes, revision requirements,
    and performance trends.
    """
    return _error_analysis_service.analyze_student_errors(student_id)


@router.get("/progress/{student_id}/history", response_model=list[AttemptRecord])
def get_history(student_id: str, limit: int = Query(default=50, ge=1, le=500)):
    """Full (paginated) practice attempt history for one student, most recent first."""
    return _progress_service.get_history(student_id, limit=limit)
