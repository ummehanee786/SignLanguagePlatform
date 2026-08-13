from fastapi import APIRouter
from app.services.progress_service import get_progress_service

router = APIRouter()

@router.get("/admin/overview")
def get_admin_overview():
    """
    Returns an aggregated view of all students for the Instructor/Trainer Dashboard.
    """
    return get_progress_service().get_cohort_overview()
