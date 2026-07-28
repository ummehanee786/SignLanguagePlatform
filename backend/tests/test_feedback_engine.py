"""
test_feedback_engine.py

Unit tests for the GestureFeedbackEngine and individual LandmarkRules.

Run with:
    cd d:\\SignLanguagePlatform\\backend
    ..\\venv\\Scripts\\python -m pytest tests/test_feedback_engine.py -v
"""

import math
import pytest

from app.ai.feedback.models import LandmarkDeviation, DetailedFeedback
from app.ai.feedback.landmark_rules import (
    FingerExtensionRule,
    FingerCrossingRule,
    ThumbUnderFingersRule,
    ThumbExtensionRule,
    PalmOrientationRule,
    LandmarkRule,
)
from app.ai.feedback.feedback_engine import GestureFeedbackEngine


# ---------------------------------------------------------------------------
# Synthetic landmark helpers
# ---------------------------------------------------------------------------

def _flat_landmarks(lm_dict: dict) -> list:
    """
    Build a 63-float flat landmark list from a dict of {id: (x, y, z)}.
    Any landmark ID not in lm_dict gets coordinates (0.5, 0.5, 0.0).
    """
    result = []
    for i in range(21):
        x, y, z = lm_dict.get(i, (0.5, 0.5, 0.0))
        result.extend([x, y, z])
    return result


def _extended_index_landmarks() -> list:
    """
    Synthetic landmarks where the index finger is clearly EXTENDED:
      - index tip (8) has lower y than index PIP (6)  →  tip is higher = extended
    All other fingers bent or neutral.
    """
    lm = {}
    # Wrist at bottom
    lm[0] = (0.5, 0.9, 0.0)
    # Index: extended (tip higher = lower y)
    lm[5] = (0.5, 0.6, 0.0)   # MCP
    lm[6] = (0.5, 0.5, 0.0)   # PIP
    lm[7] = (0.5, 0.4, 0.0)   # DIP
    lm[8] = (0.5, 0.3, 0.0)   # tip (extended)
    # Middle bent
    lm[9]  = (0.55, 0.6, 0.0)
    lm[10] = (0.55, 0.55, 0.0)
    lm[12] = (0.55, 0.6, 0.0)  # tip higher than PIP → bent
    # Ring bent
    lm[13] = (0.6, 0.6, 0.0)
    lm[14] = (0.6, 0.55, 0.0)
    lm[16] = (0.6, 0.6, 0.0)
    # Pinky bent
    lm[17] = (0.65, 0.6, 0.0)
    lm[18] = (0.65, 0.55, 0.0)
    lm[20] = (0.65, 0.6, 0.0)
    return _flat_landmarks(lm)


def _bent_index_landmarks() -> list:
    """
    Synthetic landmarks where the index finger is clearly BENT:
      - index tip (8) y > index PIP (6) y  →  tip is below PIP → bent/curled
    """
    lm = dict(_lm_pairs_from_flat(_extended_index_landmarks()))
    # Make index tip lower in image (larger y) than PIP → bent
    lm[8] = (0.5, 0.6, 0.0)   # tip now at same y as PIP — slightly below
    lm[6] = (0.5, 0.45, 0.0)  # PIP higher
    # Now tip.y (0.6) > pip.y (0.45) → bent
    return _flat_landmarks(lm)


def _lm_pairs_from_flat(flat: list) -> dict:
    """Convert a 63-float flat list back to {id: (x,y,z)} dict."""
    result = {}
    for i in range(21):
        base = i * 3
        result[i] = tuple(flat[base:base + 3])
    return result


def _crossed_fingers_landmarks() -> list:
    """Index tip to LEFT of middle tip → crossed (R shape)."""
    lm = _lm_pairs_from_flat(_extended_index_landmarks())
    lm[8]  = (0.40, 0.3, 0.0)   # index tip: LEFT
    lm[12] = (0.55, 0.3, 0.0)   # middle tip: RIGHT
    return _flat_landmarks(lm)


def _uncrossed_fingers_landmarks() -> list:
    """Index tip to RIGHT of middle tip → side-by-side (U shape)."""
    lm = _lm_pairs_from_flat(_extended_index_landmarks())
    lm[8]  = (0.55, 0.3, 0.0)   # index tip: RIGHT
    lm[12] = (0.45, 0.3, 0.0)   # middle tip: LEFT
    return _flat_landmarks(lm)


def _thumb_not_tucked_landmarks() -> list:
    """Thumb tip clearly ABOVE the PIP joints → not tucked under."""
    lm = _lm_pairs_from_flat(_extended_index_landmarks())
    # PIP joints at y=0.55; thumb tip high up (small y)
    lm[6]  = (0.5,  0.55, 0.0)   # index PIP
    lm[10] = (0.55, 0.55, 0.0)   # middle PIP
    lm[4]  = (0.4,  0.30, 0.0)   # thumb tip: well above (lower y)
    return _flat_landmarks(lm)


def _thumb_tucked_landmarks() -> list:
    """Thumb tip clearly BELOW the PIP joints → properly tucked."""
    lm = _lm_pairs_from_flat(_extended_index_landmarks())
    lm[6]  = (0.5,  0.45, 0.0)   # index PIP
    lm[10] = (0.55, 0.45, 0.0)   # middle PIP
    lm[4]  = (0.4,  0.60, 0.0)   # thumb tip: below (larger y)
    return _flat_landmarks(lm)


# ---------------------------------------------------------------------------
# GestureFeedbackEngine — top-level tests
# ---------------------------------------------------------------------------

class TestFeedbackEngineNoLandmarks:
    def test_returns_existing_message_when_no_landmarks(self):
        engine = GestureFeedbackEngine()
        fb = engine.evaluate("R", "U", landmarks=None, existing_message="Not quite.")
        assert fb.overall_message == "Not quite."
        assert fb.deviations == []
        assert fb.correction_messages == []

    def test_returns_existing_message_when_wrong_length(self):
        engine = GestureFeedbackEngine()
        fb = engine.evaluate("R", "U", landmarks=[0.1, 0.2], existing_message="Try again.")
        assert fb.overall_message == "Try again."
        assert fb.deviations == []


class TestFeedbackEngineCorrectPrediction:
    def test_no_rules_run_on_correct_prediction(self):
        """Rules should never fire when expected == predicted."""
        # Use bent finger landmarks that would normally trigger FingerExtensionRule
        # for letter D (which needs index extended); but since it's "correct",
        # no rule should run.
        engine = GestureFeedbackEngine()
        fb = engine.evaluate(
            expected="D", predicted="D",
            landmarks=_bent_index_landmarks(),
            existing_message="Correct!",
            severity="success",
        )
        assert fb.severity == "success"
        assert fb.deviations == []
        assert "Correct" in fb.overall_message


class TestFeedbackEngineRuleInjection:
    def test_custom_rule_is_called(self):
        """Custom rule injected via constructor should be evaluated."""
        class AlwaysFireRule:
            def evaluate(self, expected, predicted, landmarks):
                return LandmarkDeviation(
                    rule_name="always",
                    description="Always fires",
                    severity="warning",
                )

        engine = GestureFeedbackEngine(rules=[AlwaysFireRule()])
        fb = engine.evaluate(
            "A", "B",
            landmarks=_flat_landmarks({}),
            existing_message="Wrong.",
        )
        assert len(fb.deviations) == 1
        assert fb.deviations[0].rule_name == "always"
        assert "Always fires" in fb.correction_messages[0]

    def test_empty_rules_list_returns_existing(self):
        engine = GestureFeedbackEngine(rules=[])
        fb = engine.evaluate("A", "B", landmarks=_flat_landmarks({}), existing_message="Msg.")
        assert fb.overall_message == "Msg."
        assert fb.deviations == []


# ---------------------------------------------------------------------------
# FingerExtensionRule
# ---------------------------------------------------------------------------

class TestFingerExtensionRule:
    rule = FingerExtensionRule()

    def test_no_deviation_for_D_with_extended_index(self):
        """D requires index extended; extended index should produce no deviation."""
        result = self.rule.evaluate("D", "O", _extended_index_landmarks())
        # Middle/ring/pinky are ambiguous/bent which is correct for D
        # index extended → no deviation for index
        # Middle/ring/pinky should be bent for D — check no "should be extended" error
        if result:
            # If there's a result, it should not be about index
            assert "index finger (should be extended)" not in result.description

    def test_deviation_fires_for_B_with_bent_index(self):
        """B requires all 4 fingers extended; bent index should trigger a deviation."""
        # _bent_index_landmarks has index clearly bent
        result = self.rule.evaluate("B", "A", _bent_index_landmarks())
        assert result is not None
        assert result.rule_name == "finger_extension"
        assert "index finger" in result.description

    def test_none_for_unknown_letter(self):
        result = self.rule.evaluate("1", "2", _extended_index_landmarks())
        assert result is None


# ---------------------------------------------------------------------------
# FingerCrossingRule
# ---------------------------------------------------------------------------

class TestFingerCrossingRule:
    rule = FingerCrossingRule()

    def test_fires_for_R_expected_but_uncrossed(self):
        """Expected R but fingers are side-by-side (U shape) → should fire."""
        result = self.rule.evaluate("R", "U", _uncrossed_fingers_landmarks())
        assert result is not None
        assert result.rule_name == "finger_crossing"
        assert "CROSSED" in result.description

    def test_fires_for_U_expected_but_crossed(self):
        """Expected U but fingers are crossed (R shape) → should fire."""
        result = self.rule.evaluate("U", "R", _crossed_fingers_landmarks())
        assert result is not None
        assert result.rule_name == "finger_crossing"
        assert "side-by-side" in result.description

    def test_no_fire_for_R_correctly_crossed(self):
        """Expected R and fingers ARE crossed → no deviation."""
        result = self.rule.evaluate("R", "U", _crossed_fingers_landmarks())
        assert result is None

    def test_ignores_non_r_u_pair(self):
        """Rule only targets R/U confusion."""
        result = self.rule.evaluate("A", "B", _crossed_fingers_landmarks())
        assert result is None


# ---------------------------------------------------------------------------
# ThumbUnderFingersRule
# ---------------------------------------------------------------------------

class TestThumbUnderFingersRule:
    rule = ThumbUnderFingersRule()

    def test_fires_when_thumb_not_tucked_for_N(self):
        result = self.rule.evaluate("N", "M", _thumb_not_tucked_landmarks())
        assert result is not None
        assert result.rule_name == "thumb_under_fingers"
        assert "tucked" in result.description.lower()

    def test_no_fire_when_thumb_tucked_for_M(self):
        result = self.rule.evaluate("M", "N", _thumb_tucked_landmarks())
        assert result is None

    def test_ignores_non_m_n_letter(self):
        result = self.rule.evaluate("A", "B", _thumb_not_tucked_landmarks())
        assert result is None


# ---------------------------------------------------------------------------
# LandmarkRule protocol compliance
# ---------------------------------------------------------------------------

class TestLandmarkRuleProtocol:
    def test_all_default_rules_implement_protocol(self):
        from app.ai.feedback.landmark_rules import default_rules
        rules = default_rules()
        assert len(rules) > 0
        for rule in rules:
            assert isinstance(rule, LandmarkRule), (
                f"{type(rule).__name__} does not satisfy the LandmarkRule protocol"
            )
            assert callable(getattr(rule, "evaluate", None))
