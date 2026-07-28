"""
models.py

The two structured objects the Sign Accuracy Assessment Engine produces:

    AssessmentRecord - the full, storable result of one assessment
                        (what Task 1 calls "an Assessment Record, not
                        just a prediction")
    FeedbackObject   - a human-facing message derived from the record,
                        the final box in the SRS pipeline diagram

Both are plain dataclasses, not Pydantic models - same reasoning as
PredictionResult (app/ai/ml/inference/result.py): this is the AI
module's own internal contract, independent of the web framework. The
API layer converts these into whatever response shape it needs.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import List, Optional, Dict


@dataclass
class AssessmentRecord:
    """
    The full result of one sign-accuracy assessment. This is what gets
    persisted (via ProgressService) and is the thing Task 1 explicitly
    asks for instead of a bare prediction.
    """
    expected_gesture: str
    predicted_gesture: Optional[str]
    correct: bool
    confidence: float
    gesture_accuracy: float  # student's historical accuracy on THIS letter (including this attempt)
    attempt_number: int
    time_taken_seconds: float  # wall-clock time the student took to perform this attempt
    session_accuracy: float
    model_version: Optional[str] = None
    student_id: Optional[str] = None
    session_id: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    # --- Motion-based metrics (Task 2) - populated only when the caller
    # streamed frames via /practice/{session_id}/stream-frame before this
    # attempt; None if no capture window was available for this attempt,
    # so older/simpler callers keep working unchanged. ---
    gesture_stability: Optional[float] = None            # 0-100, higher = steadier hold
    invalid_frames_before_valid: Optional[int] = None     # frames before the hand was usable
    average_confidence_over_gesture: Optional[float] = None  # mean confidence across the capture window
    overall_sign_score: Optional[float] = None            # 0-100, combines accuracy + confidence + timing
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeedbackObject:
    """
    Human-facing feedback derived from an AssessmentRecord. Kept
    separate from AssessmentRecord itself (rather than just adding a
    "message" field to it) because a record is data to be stored and
    aggregated, while feedback is presentation - one AssessmentRecord
    could reasonably support different feedback phrasing/localization
    later without touching the stored record format.
    """
    message: str
    tip: Optional[str] = None
    severity: str = "info"  # "success" | "info" | "warning" - a UI styling hint
    # Landmark-level correction details from GestureFeedbackEngine.
    # Empty for correct predictions or when landmarks aren't available.
    deviations: List[dict] = field(default_factory=list)
    correction_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)