from fastapi import APIRouter, File, HTTPException, UploadFile

from app.content.lesson_service import LessonService
from app.services.session_service import SessionService
from app.services.gesture_service import GestureService
from app.services.progress_service import get_progress_service
from app.services.assessment_service import AssessmentService
from app.ai.assessment.sign_accuracy_engine import SignAccuracyAssessmentEngine
from app.schemas.practice import StartPracticeResponse, AttemptResponse, EndPracticeResponse
from app.schemas.session_review import SessionReviewResponse

router = APIRouter()

_lesson_service = LessonService()
_session_service = SessionService()
_gesture_service = GestureService()
_assessment_engine = SignAccuracyAssessmentEngine(get_progress_service())
assessment_service = AssessmentService(
    _lesson_service, _session_service, _gesture_service, _assessment_engine
)


@router.post("/practice/start/{lesson_id}", response_model=StartPracticeResponse)
def start_practice(lesson_id: int, student_id: str, auto_next: bool = True):
    """
    User selects an alphabet (lesson_id) and clicks "Start Practice" -
    starts a session and returns the reference sign to display.

    `student_id` identifies who's practicing (a name or client-generated
    ID - there's no login system yet, see progress_service.py).
    `auto_next` controls whether a correct answer automatically advances
    to the next letter (A -> B -> C -> ...); pass false to stay on one
    letter until the student ends the session themselves.
    """
    result = assessment_service.start_practice(lesson_id, student_id, auto_next)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Lesson {lesson_id} not found")
    return result


@router.post("/practice/{session_id}/attempt", response_model=AttemptResponse)
async def submit_attempt(session_id: str, file: UploadFile = File(...)):
    """
    Webcam captures a frame (uploaded here) -> Live Prediction -> Gesture
    Comparison -> Accuracy Score Calculation -> Assessment Result ->
    Feedback Object, via the Sign Accuracy Assessment Engine.

    If auto_next was enabled at session start and the attempt was
    correct, the session automatically advances to the next letter -
    check `next_lesson` in the response to display it.
    """
    image_bytes = await file.read()
    result = assessment_service.submit_attempt(session_id, image_bytes)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return result


@router.post("/practice/{session_id}/end", response_model=EndPracticeResponse)
def end_practice(session_id: str):
    """Ends the practice session and returns a final summary."""
    result = assessment_service.end_practice(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return result


@router.get("/practice/{session_id}/review", response_model=SessionReviewResponse)
def get_session_review(session_id: str, student_id: str):
    """
    Returns a comprehensive post-session review once a practice session ends
    (or at any point during it).  Includes:

      - overall_score, total/correct/incorrect attempt counts
      - correct_gestures / incorrect_gestures (distinct letter lists)
      - confidence_trend: per-attempt confidence values for charting
      - most_common_mistakes: top-5 (expected, predicted) error pairs by count
      - gesture_feedback: per-letter attempt stats + last feedback message/tip
      - recommended_gestures: up to 5 weakest lifetime gestures to practise next

    `student_id` must match the one used to start the session.
    """
    result = get_progress_service().get_session_review(student_id, session_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No attempts found for session '{session_id}' "
                f"and student '{student_id}'. "
                f"Submit at least one attempt before requesting a review."
            ),
        )
    return result