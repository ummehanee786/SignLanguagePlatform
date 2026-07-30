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

from typing import List, Optional, Tuple

from app.schemas.prediction import PredictionResponse
from app.ai.assessment.models import AssessmentRecord, FeedbackObject
from app.ai.feedback.feedback_engine import GestureFeedbackEngine
from app.ai.assessment.motion_metrics import (
    GestureFrameRecord,
    average_confidence,
    gesture_stability,
    invalid_frames_before_valid,
    overall_sign_score,
    unstable_frames_before_acceptance,
)

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
        motion_records: Optional[List[GestureFrameRecord]] = None,
    ) -> Tuple[AssessmentRecord, FeedbackObject]:
        """
        Precondition: `prediction.success` must be True - a failed
        prediction (no hand detected, bad image) isn't a gesture to
        compare against anything and shouldn't reach this engine at
        all; the caller (assessment_service.submit_attempt) is
        responsible for filtering those out before calling assess().

        `motion_records` is the optional per-frame capture window from
        MotionCaptureSession.records() (see
        app/services/motion_capture_service.py) - built from frames
        streamed via POST /practice/{session_id}/stream-frame WHILE the
        student held the sign, before this single graded frame was
        submitted. When absent (a caller that never streamed frames),
        the motion-based fields degrade to None/best-effort rather than
        raising - this engine still works exactly as before.
        """
        assert prediction.success, "assess() requires a successful prediction - filter failures upstream"

        predicted_gesture = prediction.predicted_class
        correct = predicted_gesture.upper() == expected_gesture.upper()

        # --- Motion-based metrics (Task 2) ---
        stability = gesture_stability(motion_records) if motion_records else None
        invalid_before_valid = (
            invalid_frames_before_valid(motion_records) if motion_records else None
        )
        avg_confidence_over_gesture = (
            average_confidence(motion_records) if motion_records else None
        )
        unstable_before_accept = (
            unstable_frames_before_acceptance(motion_records) if motion_records else None
        )
        # Overall sign score doesn't require a capture window - it always
        # has correctness/confidence/timing to work with; stability is
        # folded in only when a window was actually captured.
        sign_score = overall_sign_score(
            handshape_correct=correct,
            confidence=prediction.confidence,
            time_taken_seconds=time_taken_seconds,
            stability=stability,
        )

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
            gesture_stability=stability,
            invalid_frames_before_valid=invalid_before_valid,
            average_confidence_over_gesture=avg_confidence_over_gesture,
            unstable_frames_before_acceptance=unstable_before_accept,
            overall_sign_score=sign_score,
        )

        # Build feedback using landmarks so it can be stored with the attempt.
        feedback = self._build_feedback(
            preliminary_record,
            landmarks=prediction.landmarks,
            prediction=prediction,
            motion_records=motion_records,
        )

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
            gesture_stability=stability,
            invalid_frames_before_valid=invalid_before_valid,
            average_confidence_over_gesture=avg_confidence_over_gesture,
            unstable_frames_before_acceptance=unstable_before_accept,
            overall_sign_score=sign_score,
        )
        # Update the record's gesture_accuracy to the real value now that
        # the attempt has been recorded and counted.
        record.gesture_accuracy = gesture_accuracy
        return record, feedback

    def _build_feedback(
        self,
        record: AssessmentRecord,
        landmarks: Optional[list] = None,
        prediction: Optional[PredictionResponse] = None,
        motion_records: Optional[List[GestureFrameRecord]] = None,
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

        extra_messages = []
        extra_tips = []

        # Rule-based feedback layer for pipeline visibility
        if prediction is not None:
            if hasattr(prediction, "has_person") and not prediction.has_person:
                extra_messages.append("No person detected in the frame.")
                extra_tips.append("Make sure you are positioned in front of the camera.")
            elif hasattr(prediction, "upper_body_visible") and not prediction.upper_body_visible:
                extra_messages.append("Please keep your upper body fully visible inside the camera frame.")
                extra_tips.append("Position yourself so your upper body (shoulders, chest) is visible.")

            if hasattr(prediction, "hand_count") and prediction.hand_count > 2:
                extra_messages.append("Only one person should be present in the camera view.")
                extra_tips.append("Ensure only a single person's hands are shown in the frame.")

            if hasattr(prediction, "partial_hand_visible") and prediction.partial_hand_visible:
                extra_messages.append("Please keep your entire hand visible within the camera view.")
                extra_tips.append("Ensure your signing hand stays completely inside the frame boundary.")
            elif hasattr(prediction, "hand_centered") and not prediction.hand_centered:
                extra_messages.append("Move your hand closer to the center of the frame.")
                extra_tips.append("Move your hand so it is positioned more centrally in the video frame.")

        # Rule-based feedback layer for motion and stability metrics
        if motion_records:
            # Check for early release: had a stable prediction at some point, but the last frames are invalid
            had_stable = any(r.valid for r in motion_records)
            last_few_valid = [r.valid for r in motion_records[-3:]]
            if had_stable and len(last_few_valid) >= 3 and not any(last_few_valid):
                extra_messages.append("Hold the gesture slightly longer before releasing.")
                extra_tips.append("Avoid dropping your hand too quickly after stabilizing.")

            # Check if hand moved/wobbled before prediction stabilized
            if record.unstable_frames_before_acceptance is not None and record.unstable_frames_before_acceptance > 8:
                extra_messages.append("Your hand moved before prediction stabilized.")
                extra_tips.append("Try to hold your hand still from the moment you start the sign.")
            elif record.gesture_stability is not None and record.gesture_stability < 65.0:
                extra_messages.append("Your hand was moving or wobbling too much.")
                extra_tips.append("Ensure you hold the pose key landmarks steady.")

        # Merge with detailed feedback
        final_message = detailed.overall_message
        if extra_messages:
            final_message += " " + " ".join(extra_messages)

        final_tip = detailed.tip
        if extra_tips:
            if final_tip:
                final_tip += " " + " ".join(extra_tips)
            else:
                final_tip = " ".join(extra_tips)

        # Retrieve personalized data and inject into messages
        tutor_notes = []
        recommendations = []
        student_id = record.student_id

        if student_id:
            try:
                from app.services.error_analysis_service import ErrorAnalysisService
                ea_service = ErrorAnalysisService(self._progress_service)
                insights = ea_service.analyze_student_errors(student_id)

                expected = record.expected_gesture.upper()
                predicted = record.predicted_gesture.upper() if record.predicted_gesture else None

                # 1. Check for repeated mistakes / common confuses
                if not record.correct and predicted:
                    is_frequent_confusion = any(
                        p.expected.upper() == expected and p.predicted.upper() == predicted
                        for p in insights.most_confused_pairs
                    )
                    if is_frequent_confusion:
                        tutor_notes.append(
                            f"You frequently confuse '{expected}' with '{predicted}'."
                        )

                    rep_mistakes = [m for m in insights.repeated_mistakes if m.alphabet.upper() == expected]
                    if rep_mistakes:
                        tutor_notes.append(
                            f"You have struggled with '{expected}' across {len(rep_mistakes[0].sessions)} different sessions."
                        )

                # 2. Check for low confidence alerts
                low_conf = [c for c in insights.low_confidence_alphabets if c.alphabet.upper() == expected]
                if low_conf:
                    tutor_notes.append(
                        f"Your average confidence on '{expected}' is low ({low_conf[0].average_confidence:.0%})."
                    )

                # 3. Check for performance trends (improvement/decline)
                trend = [t for t in insights.performance_trends if t.alphabet.upper() == expected]
                if trend:
                    if trend[0].trend == "improving":
                        tutor_notes.append(
                            f"Great job! You are improving on '{expected}' ({trend[0].earlier_accuracy:.0f}% -> {trend[0].later_accuracy:.0f}% accuracy)."
                        )
                    elif trend[0].trend == "declining":
                        tutor_notes.append(
                            f"Your recent accuracy on '{expected}' has dipped ({trend[0].earlier_accuracy:.0f}% -> {trend[0].later_accuracy:.0f}%)."
                        )

                # 4. Generate highly relevant recommendations
                # Suggest immediate revisions first (up to 2), then low confidence alphabets (up to 2)
                for gesture in insights.revision_required_gestures[:2]:
                    recommendations.append(f"Revise '{gesture}' (needs immediate review)")
                for item in insights.low_confidence_alphabets:
                    if item.alphabet not in insights.revision_required_gestures[:2] and len(recommendations) < 3:
                        recommendations.append(f"Practice '{item.alphabet}' to build confidence")

                # If we need more recommendations, fill them with weakest alphabets
                if len(recommendations) < 3:
                    dashboard = self._progress_service.get_dashboard(student_id)
                    for item in dashboard.get("weakest_alphabets", []):
                        rec_str = f"Practice '{item['alphabet']}' (historical accuracy: {item['accuracy_percentage']:.0f}%)"
                        if all(item["alphabet"] not in r for r in recommendations) and len(recommendations) < 3:
                            recommendations.append(rec_str)

                # If still empty, add a default message
                if not recommendations:
                    recommendations.append("Continue practicing the next letters in the lesson!")

            except Exception:
                pass

        if tutor_notes:
            final_message += " " + " ".join(tutor_notes)

        # Update severity to warning if critical issues are detected
        final_severity = detailed.severity
        if any(msg in final_message for msg in ["No person detected", "upper body fully visible", "Only one person"]):
            final_severity = "warning"

        return FeedbackObject(
            message=final_message,
            tip=final_tip,
            severity=final_severity,
            deviations=[d.to_dict() if hasattr(d, "to_dict") else d for d in detailed.deviations],
            correction_messages=detailed.correction_messages + extra_messages,
            recommendations=recommendations,
        )