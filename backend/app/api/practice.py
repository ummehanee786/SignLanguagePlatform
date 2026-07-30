from fastapi import APIRouter, File, HTTPException, UploadFile

from app.content.lesson_service import LessonService
from app.services.session_service import SessionService
from app.services.gesture_service import GestureService
from app.services.progress_service import get_progress_service
from app.services.assessment_service import AssessmentService
from app.services.motion_capture_service import get_motion_capture_manager
from app.ai.assessment.sign_accuracy_engine import SignAccuracyAssessmentEngine
from app.schemas.practice import StartPracticeResponse, AttemptResponse, EndPracticeResponse
from app.schemas.session_review import SessionReviewResponse
from app.schemas.stream_frame import StreamFrameResponse, SequenceResponse

router = APIRouter()

_lesson_service = LessonService()
_session_service = SessionService()
_gesture_service = GestureService()
_assessment_engine = SignAccuracyAssessmentEngine(get_progress_service())
_motion_capture_manager = get_motion_capture_manager()
assessment_service = AssessmentService(
    _lesson_service, _session_service, _gesture_service, _assessment_engine,
    _motion_capture_manager,
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


@router.post("/practice/{session_id}/stream-frame", response_model=StreamFrameResponse)
async def stream_frame(session_id: str, file: UploadFile = File(...)):
    """
    Task 1: continuous webcam capture. The client is expected to call
    this repeatedly (e.g. every ~100ms) WHILE the student holds a sign,
    BEFORE calling /attempt to actually grade it - each call is one
    webcam frame.

    Each frame runs through the exact same detection + prediction
    pipeline /attempt uses (GestureService) and is folded into this
    session's rolling window of the last 20-30 frames (oldest
    automatically discarded once full) - both the raw landmark
    sequence (exposed via GET /practice/{session_id}/sequence, for any
    future temporal model) and the per-frame validity/confidence log
    Task 2's motion-based metrics are computed from when /attempt is
    next called.

    Frames with no usable hand detection are still counted (so Task
    2's "invalid frames before a valid prediction" can see them) but
    are never added to the landmark sequence itself.
    """
    session = _session_service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    image_bytes = await file.read()
    prediction = _gesture_service.predict(image_bytes)

    capture = _motion_capture_manager.get_or_create(session_id)
    capture.add_prediction(prediction)

    return {
        "session_id": session_id,
        "buffered_frames": len(capture),
        "buffer_full": capture.is_full(),
        "frame_valid": prediction.success,
        "predicted_class": prediction.predicted_class,
        "confidence": prediction.confidence,
        "stable_prediction": capture.stable_prediction,
        "stable_confidence": capture.stable_confidence,
        "stable_streak": capture.stable_streak,
        "inference_latency": capture.latest_latency,
        "processing_fps": capture.get_processing_fps(),
        "has_person": prediction.has_person,
        "hand_count": prediction.hand_count,
        "upper_body_visible": prediction.upper_body_visible,
        "partial_hand_visible": prediction.partial_hand_visible,
        "hand_centered": prediction.hand_centered,
    }


@router.get("/practice/{session_id}/sequence", response_model=SequenceResponse)
def get_sequence(session_id: str):
    """
    Task 1 deliverable: exposes the CURRENT buffered landmark sequence
    for this session, in the exact shape a future temporal model
    (LSTM/GRU/Transformer) would be called with - a list of frames,
    each a 63-value normalized landmark vector, oldest first.

    Returns an empty sequence (not a 404) if nothing has been streamed
    yet for this session - "hasn't started capturing" isn't an error,
    just an empty window.
    """
    capture = _motion_capture_manager.get(session_id)
    if capture is None:
        return {
            "session_id": session_id,
            "frame_count": 0,
            "max_frames": 0,
            "is_full": False,
            "sequence": [],
        }

    frames = capture.sequence_buffer.frames()
    return {
        "session_id": session_id,
        "frame_count": len(frames),
        "max_frames": capture.sequence_buffer.max_frames,
        "is_full": capture.is_full(),
        "sequence": [frame.tolist() for frame in frames],
    }


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