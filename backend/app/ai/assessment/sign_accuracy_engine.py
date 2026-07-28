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

Sprint update: the engine now also calls GestureFeedbackEngine to perform
landmark-level deviation analysis (FingerExtension, ThumbExtension, etc.)
and enriches the FeedbackObject with structured correction messages.
"""

from typing import Optional, Tuple

from app.schemas.prediction import PredictionResponse
from app.ai.assessment.models import AssessmentRecord, FeedbackObject
from app.ai.feedback.feedback_engine import GestureFeedbackEngine

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

    From this sprint: also calls GestureFeedbackEngine to enrich the
    FeedbackObject with landmark-level deviation corrections.
    """

    def __init__(self, progress_service):
        self._progress_service = progress_service
        self._feedback_engine = GestureFeedbackEngine()

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

        # Build a preliminary AssessmentRecord (without final gesture_accuracy
        # yet — we compute that after recording) so _build_feedback has all
        # the data it needs but gesture_accuracy is a placeholder.
        preliminary_record = AssessmentRecord(
            expected_gesture=expected_gesture.upper(),
            predicted_gesture=predicted_gesture.upper(),
            correct=correct,
            confidence=round(prediction.confidence, 4),
            gesture_accuracy=0.0,   # placeholder; updated below
            attempt_number=attempt_number,
            time_taken_seconds=round(time_taken_seconds, 2),
            session_accuracy=session_accuracy,
            model_version=prediction.model_version,
            student_id=student_id,
            session_id=session_id,
            probabilities=prediction.probabilities,
        )

        # Build feedback using landmarks so it can be stored with the attempt.
        feedback = self._build_feedback(preliminary_record, landmarks=prediction.landmarks)

        # Store the attempt (including feedback message) so gesture_accuracy
        # below reflects this attempt too.
        self._progress_service.record_attempt(
            student_id=student_id,
            alphabet_practiced=expected_gesture,
            predicted_alphabet=predicted_gesture,
            correct=correct,
            confidence=prediction.confidence,
            inference_time=prediction.processing_time,
            time_taken_seconds=time_taken_seconds,
            session_id=session_id,
            feedback_message=feedback.message,
            feedback_tip=feedback.tip,
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
        # Update the record's gesture_accuracy to the real value now that
        # the attempt has been recorded and counted.
        record.gesture_accuracy = gesture_accuracy
        return record, feedback

    def _build_feedback(
        self,
        record: AssessmentRecord,
        landmarks: Optional[list] = None,
    ) -> FeedbackObject:
        motion_note = (
            f" Note: '{record.expected_gesture}' is traditionally a motion-based "
            f"sign in ASL, but this assessment only sees a single still frame - "
            f"treat this result as a rough check, not a definitive one."
            if record.expected_gesture in _MOTION_LETTERS
            else ""
        )

        # --- Build base message and tip (same logic as before) ---
        if record.correct:
            if record.confidence >= HIGH_CONFIDENCE_THRESHOLD:
                base_message = f"Correct! Your '{record.expected_gesture}' was recognized clearly."
                base_tip = motion_note or None
                severity = "success"
            else:
                base_message = (
                    f"Correct, but the model wasn't fully confident "
                    f"({record.confidence:.0%})."
                )
                base_tip = (
                    "Try holding the sign a bit more clearly, with your whole "
                    "hand visible and well-lit." + motion_note
                )
                severity = "success"
        else:
            pair_tip = None
            if record.predicted_gesture:
                pair_tip = _CONFUSABLE_PAIR_TIPS.get(
                    frozenset({record.expected_gesture, record.predicted_gesture})
                )

            base_message = (
                f"Not quite - expected '{record.expected_gesture}' but "
                f"detected '{record.predicted_gesture}'."
            )
            if pair_tip:
                base_tip = pair_tip + motion_note
            else:
                base_tip = (
                    f"Review the reference image for '{record.expected_gesture}' "
                    f"and try again." + motion_note
                )
            severity = "warning"

        # --- Enrich with landmark-level deviations via GestureFeedbackEngine ---
        detailed = self._feedback_engine.evaluate(
            expected=record.expected_gesture,
            predicted=record.predicted_gesture or "",
            landmarks=landmarks,
            existing_message=base_message,
            existing_tip=base_tip,
            severity=severity,
        )

        return FeedbackObject(
            message=detailed.overall_message,
            tip=detailed.tip,
            severity=detailed.severity,
            deviations=[d.to_dict() if hasattr(d, "to_dict") else d for d in detailed.deviations],
            correction_messages=detailed.correction_messages,
        )