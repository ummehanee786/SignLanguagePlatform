from fastapi import APIRouter
from app.services.certification_service import get_certification_service

router = APIRouter(prefix="/certifications", tags=["certifications"])


@router.get("/{student_id}")
def get_certifications(student_id: str):
    """
    Return all previously earned certifications for a student.
    Does NOT re-evaluate eligibility – fast, read-only.
    """
    return get_certification_service().get_certifications(student_id)


@router.post("/{student_id}/evaluate")
def evaluate_certifications(student_id: str):
    """
    Evaluate the student's current history against all certification levels.
    Awards any newly earned levels and returns the full status breakdown.
    """
    return get_certification_service().evaluate(student_id)
