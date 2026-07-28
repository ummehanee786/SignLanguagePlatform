"""
sign_accuracy_engine.py

The Sign Accuracy Assessment Engine described in the SRS. Implements the
pipeline:

    Expected Letter -> Live Prediction -> Gesture Comparison ->
    Accuracy Score Calculation -> Assessment Result -> Feedback Object

"Live Prediction" already exists (app/ai/ml/inference/engine.py) and is
NOT this engine's job - this engine starts from an already-made
prediction and turns it into a proper Assessment Record, not just a
bare prediction:

    Prediction = A, Confidence = 94%          <- what we had before
    -> AssessmentRecord + FeedbackObject       <- what this engine produces

This keeps Live Prediction and Gesture Comparison as separate,
independently-testable concerns, matching the SRS diagram's own
separation of boxes.
"""

from typing import Optional, Tuple

from app.schemas.prediction import PredictionResponse
from app.ai.assessment.models import AssessmentRecord, FeedbackObject

# Letter pairs known (from error_analysis.md, Task 4 of the earlier ML
# study on this exact trained model) to be genuinely hard to
# distinguish - not data-quality issues, real handshape similarity.
# Used to give specific, useful feedback instead of a generic
# "try again" when a mistake matches a known pattern.
_CONFUSABLE_PAIR_TIPS = {
    frozenset({"M", "N"}): (
        "M and N are the two closest handshapes in ASL - both are a fist "
        "with the thumb tucked under the fingers. M tucks the thumb under "
        "three fingers, N tucks it under two. Check your thumb placement."
    ),
    frozenset({"R", "U"}): (
        "R and U use the same two extended fingers (index + middle). U "
        "keeps them straight and together; R crosses them. Make sure the "
        "crossing is clearly visible to the camera."
    ),
    frozenset({"D", "O"}): (
        "D and O both use a thumb-to-finger circle shape. D fully extends "
        "the index finger; O curls it into the circle too. Check how far "
        "your index finger is extended."
    ),
}

# J and Z are motion-based letters in real ASL, but this system assesses
# a single static frame - see docs on lesson_service.py / error_analysis.md.
_MOTION_LETTERS = {"J", "Z"}

HIGH_CONFIDENCE_THRESHOLD = 0.85


class SignAccuracyAssessmentEngine:
    """
    Takes an already-made prediction (Live Prediction has already
    happened) and performs Gesture Comparison, Accuracy Score
    Calculation, and produces the Assessment Result + Feedback Object.

    Also responsible for STORING the result as an Assessment Record
    (via ProgressService) - "store the result as an Assessment Record,
    not just a prediction" is exactly what record_attempt() +
    get_gesture_accuracy() together accomplish.
    """

    def __init__(self, progress_service):
        self._progress_service = progress_service

    def assess(
        self,
        student_id: str,
        session_id: Optional[str],
        expected_gesture: str,
        prediction: PredictionResponse,
        attempt_number: int,
        time_taken_seconds: float,
        session_accuracy: float,
    ) -> Tuple[AssessmentRecord, FeedbackObject]:
        """
        Precondition: `prediction.success` must be True - a failed
        prediction (no hand detected, bad image) isn't a gesture to
        compare against anything and shouldn't reach this engine at
        all; the caller (assessment_service.submit_attempt) is
        responsible for filtering those out before calling assess().
        """
        assert prediction.success, "assess() requires a successful prediction - filter failures upstream"

        predicted_gesture = prediction.predicted_class
        correct = predicted_gesture.upper() == expected_gesture.upper()

        # Store the attempt FIRST, so gesture_accuracy below reflects
        # this attempt too (a student's dashboard/report should always
        # be able to say "as of your most recent attempt, here's how
        # you're doing on this letter").
        self._progress_service.record_attempt(
            student_id=student_id,
            alphabet_practiced=expected_gesture,
            predicted_alphabet=predicted_gesture,
            correct=correct,
            confidence=prediction.confidence,
            inference_time=prediction.processing_time,
            time_taken_seconds=time_taken_seconds,
            session_id=session_id,
        )
        gesture_accuracy = self._progress_service.get_gesture_accuracy(student_id, expected_gesture)

        record = AssessmentRecord(
            expected_gesture=expected_gesture.upper(),
            predicted_gesture=predicted_gesture.upper(),
            correct=correct,
            confidence=round(prediction.confidence, 4),
            gesture_accuracy=gesture_accuracy,
            attempt_number=attempt_number,
            time_taken_seconds=round(time_taken_seconds, 2),
            session_accuracy=session_accuracy,
            model_version=prediction.model_version,
            student_id=student_id,
            session_id=session_id,
            probabilities=prediction.probabilities,
        )

        feedback = self._build_feedback(record)
        return record, feedback

    def _build_feedback(self, record: AssessmentRecord) -> FeedbackObject:
        motion_note = (
            f" Note: '{record.expected_gesture}' is traditionally a motion-based "
            f"sign in ASL, but this assessment only sees a single still frame - "
            f"treat this result as a rough check, not a definitive one."
            if record.expected_gesture in _MOTION_LETTERS
            else ""
        )

        if record.correct:
            if record.confidence >= HIGH_CONFIDENCE_THRESHOLD:
                return FeedbackObject(
                    message=f"Correct! Your '{record.expected_gesture}' was recognized clearly.",
                    severity="success",
                    tip=motion_note or None,
                )
            return FeedbackObject(
                message=(
                    f"Correct, but the model wasn't fully confident "
                    f"({record.confidence:.0%})."
                ),
                tip=("Try holding the sign a bit more clearly, with your whole "
                     "hand visible and well-lit." + motion_note),
                severity="success",
            )

        pair_tip = None
        if record.predicted_gesture:
            pair_tip = _CONFUSABLE_PAIR_TIPS.get(
                frozenset({record.expected_gesture, record.predicted_gesture})
            )

        if pair_tip:
            return FeedbackObject(
                message=(
                    f"Not quite - expected '{record.expected_gesture}' but "
                    f"detected '{record.predicted_gesture}'."
                ),
                tip=pair_tip + motion_note,
                severity="warning",
            )

        return FeedbackObject(
            message=(
                f"Not quite - expected '{record.expected_gesture}' but "
                f"detected '{record.predicted_gesture}'."
            ),
            tip=(f"Review the reference image for '{record.expected_gesture}' "
                 f"and try again." + motion_note),
            severity="warning",
        )