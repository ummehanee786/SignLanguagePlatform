from fastapi import APIRouter, Query
from app.services.notification_service import get_notification_service

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("/{student_id}")
def get_notifications(
    student_id: str,
    unread_only: bool = Query(False, description="Return only unread notifications"),
):
    """Fetch all stored notifications for a student."""
    return get_notification_service().get_notifications(student_id, unread_only=unread_only)


@router.post("/{student_id}/generate")
def generate_notifications(student_id: str):
    """
    Evaluate the student's history and push any new contextual notifications.
    Safe to call after each practice session or on page load.
    """
    added = get_notification_service().generate(student_id)
    return {"generated": len(added), "notifications": added}


@router.patch("/{student_id}/{notification_id}/read")
def mark_read(student_id: str, notification_id: str):
    """Mark a single notification as read."""
    found = get_notification_service().mark_read(student_id, notification_id)
    return {"success": found}


@router.patch("/{student_id}/read-all")
def mark_all_read(student_id: str):
    """Mark ALL notifications for a student as read."""
    count = get_notification_service().mark_all_read(student_id)
    return {"marked_read": count}
