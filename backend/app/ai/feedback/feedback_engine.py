"""
feedback_engine.py  (app/ai/feedback/)

GestureFeedbackEngine: the orchestrator that applies landmark rules and
produces a DetailedFeedback object.

Usage
-----
    from app.ai.feedback.feedback_engine import GestureFeedbackEngine

    engine = GestureFeedbackEngine()          # uses default ASL rules
    feedback = engine.evaluate(
        expected="R",
        predicted="U",
        landmarks=prediction.landmarks,       # list[float] | None
        existing_message="Not quite – expected 'R' but detected 'U'.",
        existing_tip="Review the reference image for 'R' and try again.",
        severity="warning",
    )

Design notes
------------
- Rules are evaluated in the order they appear in self._rules.
- Every non-None result is collected as a LandmarkDeviation.
- If landmarks is None (no hand was usefully detected, or the inference
  engine didn't provide them), the engine returns a DetailedFeedback
  with no deviations and just the existing message/tip passed in.
- For CORRECT predictions no deviation analysis is run — the engine
  immediately returns a success-flavoured DetailedFeedback.
- The rule list is injected via the constructor, making individual
  rules trivially testable without constructing the whole engine.
"""

from typing import Optional

from app.ai.feedback.models import DetailedFeedback, LandmarkDeviation
from app.ai.feedback.landmark_rules import default_rules, LandmarkRule


class GestureFeedbackEngine:
    """
    Applies a sequence of LandmarkRule objects to detected hand landmarks
    and returns a DetailedFeedback describing corrections the student
    should make.
    """

    def __init__(self, rules: Optional[list] = None):
        # Accept None or an explicit list (for tests / custom rule sets)
        self._rules: list[LandmarkRule] = rules if rules is not None else default_rules()

    def evaluate(
        self,
        expected: str,
        predicted: str,
        landmarks: Optional[list],
        existing_message: str = "",
        existing_tip: Optional[str] = None,
        severity: str = "warning",
    ) -> DetailedFeedback:
        """
        Evaluate landmark rules for the given prediction and return
        a DetailedFeedback.

        Parameters
        ----------
        expected        : the letter the student was supposed to sign
        predicted       : the letter the model detected
        landmarks       : flat list of 63 floats from MediaPipe (or None)
        existing_message: the message already produced by the assessment
                          engine (used as overall_message when rules
                          produce no additional detail)
        existing_tip    : the tip already produced by the assessment engine
        severity        : "success" | "info" | "warning"

        Returns
        -------
        DetailedFeedback — always, never raises
        """
        # Correct prediction: rule evaluation is meaningless (there's nothing
        # to correct), just wrap the existing message.
        if expected.upper() == predicted.upper():
            return DetailedFeedback(
                overall_message=existing_message,
                severity=severity,
                tip=existing_tip,
                deviations=[],
                correction_messages=[],
            )

        # No landmarks available: return existing feedback untouched.
        if not landmarks or len(landmarks) != 63:
            return DetailedFeedback(
                overall_message=existing_message,
                severity=severity,
                tip=existing_tip,
                deviations=[],
                correction_messages=[],
            )

        # Run every rule and collect deviations.
        deviations: list[LandmarkDeviation] = []
        try:
            for rule in self._rules:
                result = rule.evaluate(expected, predicted, landmarks)
                if result is not None:
                    deviations.append(result)
        except Exception:
            # Rules must never crash the prediction pipeline —
            # if a rule unexpectedly errors, silently skip its output.
            pass

        correction_messages = [d.description for d in deviations]

        # If rules produced corrections, compose an enriched overall message.
        if correction_messages:
            overall = (
                f"{existing_message} "
                f"Here's what to adjust: {correction_messages[0]}"
                if existing_message
                else correction_messages[0]
            )
        else:
            overall = existing_message

        return DetailedFeedback(
            overall_message=overall,
            severity=severity,
            tip=existing_tip,
            deviations=deviations,
            correction_messages=correction_messages,
        )
