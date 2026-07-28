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
from typing import Optional, Dict


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

    def to_dict(self) -> dict:
        return asdict(self)