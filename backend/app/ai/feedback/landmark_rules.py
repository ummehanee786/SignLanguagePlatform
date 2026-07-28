"""
landmark_rules.py  (app/ai/feedback/)

ASL-specific landmark deviation rules for the Gesture Feedback Engine.

Design principles
-----------------
1. Each rule is a standalone class implementing the LandmarkRule protocol.
   No rule knows about any other rule.
2. To add a new rule, create a new class and add it to DEFAULT_RULES — no
   existing rule needs to change (open/closed principle).
3. Rules operate on the raw 21-landmark flat array produced by MediaPipe
   (63 floats: x0,y0,z0 … x20,y20,z20).  Coordinates are normalised
   [0,1] relative to the image frame (x increases right, y increases
   downward).

MediaPipe Hand Landmark IDs (used below)
-----------------------------------------
 0  Wrist
 1  Thumb CMC     2  Thumb MCP     3  Thumb IP    4  Thumb tip
 5  Index MCP     6  Index PIP     7  Index DIP   8  Index tip
 9  Middle MCP   10  Middle PIP   11  Middle DIP  12  Middle tip
13  Ring MCP     14  Ring PIP     15  Ring DIP    16  Ring tip
17  Pinky MCP    18  Pinky PIP    19  Pinky DIP   20  Pinky tip
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from app.ai.feedback.models import LandmarkDeviation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lm(landmarks: list[float], idx: int) -> tuple[float, float, float]:
    """Return (x, y, z) for landmark index idx."""
    base = idx * 3
    return landmarks[base], landmarks[base + 1], landmarks[base + 2]


def _dist2d(a: tuple, b: tuple) -> float:
    """Euclidean 2D distance between two (x, y, …) tuples."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LandmarkRule(Protocol):
    """
    Every rule must expose an `evaluate` method:

        evaluate(expected, predicted, landmarks) -> LandmarkDeviation | None

    Return None if the rule finds nothing wrong; a LandmarkDeviation if it
    detects a correctable issue. Rules should be stateless.
    """
    def evaluate(
        self,
        expected: str,
        predicted: str,
        landmarks: list[float],
    ) -> Optional[LandmarkDeviation]:
        ...


# ---------------------------------------------------------------------------
# Concrete rules
# ---------------------------------------------------------------------------

# Mapping: letter -> which of the 4 fingers (index/middle/ring/pinky) should
# be EXTENDED for that ASL static letter.
# Each finger is represented as (tip_id, pip_id).
# True  = finger should be clearly extended (tip above pip in image, i.e. tip.y < pip.y)
# False = finger should be bent/closed
_FINGER_EXTENSION_MAP: dict[str, dict[str, bool]] = {
    # Key: letter | Values: {"index": bool, "middle": bool, "ring": bool, "pinky": bool}
    "A": {"index": False, "middle": False, "ring":  False, "pinky": False},
    "B": {"index": True,  "middle": True,  "ring":  True,  "pinky": True},
    "C": {"index": False, "middle": False, "ring":  False, "pinky": False},  # all slightly curled
    "D": {"index": True,  "middle": False, "ring":  False, "pinky": False},
    "E": {"index": False, "middle": False, "ring":  False, "pinky": False},
    "F": {"index": False, "middle": True,  "ring":  True,  "pinky": True},
    "G": {"index": True,  "middle": False, "ring":  False, "pinky": False},
    "H": {"index": True,  "middle": True,  "ring":  False, "pinky": False},
    "I": {"index": False, "middle": False, "ring":  False, "pinky": True},
    "J": {"index": False, "middle": False, "ring":  False, "pinky": True},  # motion letter
    "K": {"index": True,  "middle": True,  "ring":  False, "pinky": False},
    "L": {"index": True,  "middle": False, "ring":  False, "pinky": False},
    "M": {"index": False, "middle": False, "ring":  False, "pinky": False},
    "N": {"index": False, "middle": False, "ring":  False, "pinky": False},
    "O": {"index": False, "middle": False, "ring":  False, "pinky": False},
    "P": {"index": True,  "middle": True,  "ring":  False, "pinky": False},
    "Q": {"index": True,  "middle": False, "ring":  False, "pinky": False},
    "R": {"index": True,  "middle": True,  "ring":  False, "pinky": False},
    "S": {"index": False, "middle": False, "ring":  False, "pinky": False},
    "T": {"index": False, "middle": False, "ring":  False, "pinky": False},
    "U": {"index": True,  "middle": True,  "ring":  False, "pinky": False},
    "V": {"index": True,  "middle": True,  "ring":  False, "pinky": False},
    "W": {"index": True,  "middle": True,  "ring":  True,  "pinky": False},
    "X": {"index": False, "middle": False, "ring":  False, "pinky": False},  # hook shape
    "Y": {"index": False, "middle": False, "ring":  False, "pinky": True},
    "Z": {"index": True,  "middle": False, "ring":  False, "pinky": False},  # motion letter
}

_FINGER_LANDMARKS = {
    "index":  (8,  6),   # (tip_id, pip_id)
    "middle": (12, 10),
    "ring":   (16, 14),
    "pinky":  (20, 18),
}

_FINGER_NAMES = {
    "index":  "index finger",
    "middle": "middle finger",
    "ring":   "ring finger",
    "pinky":  "pinky",
}

# Threshold: if |tip.y - pip.y| < EXTENSION_TOLERANCE in normalised coords,
# the finger is considered "ambiguous" and the rule won't fire — it only
# fires on clearly wrong shapes.
_EXTENSION_TOLERANCE = 0.04


class FingerExtensionRule:
    """
    Checks whether each finger's extended/bent state matches what the
    expected ASL letter requires, using (tip.y vs pip.y):
      - tip.y < pip.y  →  extended  (tip is higher in image = lower y value)
      - tip.y > pip.y  →  bent
    """

    def evaluate(
        self,
        expected: str,
        predicted: str,
        landmarks: list[float],
    ) -> Optional[LandmarkDeviation]:
        expected_map = _FINGER_EXTENSION_MAP.get(expected.upper())
        if expected_map is None:
            return None  # unknown letter — skip

        wrong_fingers = []
        for finger, (tip_id, pip_id) in _FINGER_LANDMARKS.items():
            should_extend = expected_map[finger]
            tip = _lm(landmarks, tip_id)
            pip = _lm(landmarks, pip_id)
            diff = pip[1] - tip[1]  # positive → extended, negative → bent

            if abs(diff) < _EXTENSION_TOLERANCE:
                continue  # ambiguous, skip

            is_extended = diff > 0
            if should_extend and not is_extended:
                wrong_fingers.append(f"{_FINGER_NAMES[finger]} (should be extended)")
            elif not should_extend and is_extended:
                wrong_fingers.append(f"{_FINGER_NAMES[finger]} (should be bent/closed)")

        if not wrong_fingers:
            return None

        fingers_str = ", ".join(wrong_fingers)
        return LandmarkDeviation(
            rule_name="finger_extension",
            description=(
                f"Finger position mismatch for '{expected}': {fingers_str}. "
                f"Check which fingers should be extended vs. closed for this letter."
            ),
            severity="error",
        )


class ThumbExtensionRule:
    """
    Checks whether the thumb is appropriately extended or tucked for the
    expected letter, using the thumb tip's distance from the wrist
    relative to the index MCP.
    """

    # Letters where the thumb should clearly be extended (tip far from wrist)
    _THUMB_EXTENDED: set[str] = {"A", "C", "D", "G", "H", "L", "Q", "Y"}
    # Letters where the thumb should be tucked / close to the fist
    _THUMB_TUCKED: set[str] = {"E", "M", "N", "S", "T"}

    def evaluate(
        self,
        expected: str,
        predicted: str,
        landmarks: list[float],
    ) -> Optional[LandmarkDeviation]:
        letter = expected.upper()
        if letter not in self._THUMB_EXTENDED and letter not in self._THUMB_TUCKED:
            return None

        wrist = _lm(landmarks, 0)
        index_mcp = _lm(landmarks, 5)
        thumb_tip = _lm(landmarks, 4)

        # Normalise by wrist-to-index-MCP distance so this is scale-independent
        ref_dist = _dist2d(wrist, index_mcp)
        if ref_dist < 1e-6:
            return None  # degenerate landmarks

        thumb_dist = _dist2d(wrist, thumb_tip) / ref_dist

        if letter in self._THUMB_EXTENDED and thumb_dist < 0.8:
            return LandmarkDeviation(
                rule_name="thumb_extension",
                description=(
                    f"For '{letter}', the thumb should be clearly extended/visible, "
                    f"but it appears tucked in. Extend your thumb outward."
                ),
                severity="warning",
            )
        if letter in self._THUMB_TUCKED and thumb_dist > 1.4:
            return LandmarkDeviation(
                rule_name="thumb_extension",
                description=(
                    f"For '{letter}', the thumb should be tucked in or close to the fist, "
                    f"but it appears extended. Curl your thumb inward."
                ),
                severity="warning",
            )
        return None


class FingerCrossingRule:
    """
    Specifically targets the R vs U confusion.
    R: index and middle fingers are CROSSED (index tip.x < middle tip.x when
       palm faces camera, roughly).
    U: index and middle fingers are straight and SIDE BY SIDE (index tip.x
       is to the right of middle tip.x for a right hand facing camera).

    Only fires when expected is R or U and predicted is the other one.
    """

    def evaluate(
        self,
        expected: str,
        predicted: str,
        landmarks: list[float],
    ) -> Optional[LandmarkDeviation]:
        pair = {expected.upper(), predicted.upper()}
        if pair != {"R", "U"}:
            return None

        index_tip = _lm(landmarks, 8)
        middle_tip = _lm(landmarks, 12)
        diff_x = index_tip[0] - middle_tip[0]

        # crossed: index tip is to the LEFT of middle tip (lower x)
        # side-by-side: index tip is to the RIGHT (higher x)
        crossed = diff_x < -0.02

        if expected.upper() == "R" and not crossed:
            return LandmarkDeviation(
                rule_name="finger_crossing",
                description=(
                    "For 'R', your index and middle fingers should be clearly CROSSED. "
                    "Your fingers appear side-by-side (which looks like 'U'). "
                    "Cross the index finger over the middle finger."
                ),
                severity="error",
            )
        if expected.upper() == "U" and crossed:
            return LandmarkDeviation(
                rule_name="finger_crossing",
                description=(
                    "For 'U', your index and middle fingers should be straight and side-by-side. "
                    "Your fingers appear crossed (which looks like 'R'). "
                    "Keep both fingers straight without crossing."
                ),
                severity="error",
            )
        return None


class ThumbUnderFingersRule:
    """
    Specifically targets the M vs N confusion.
    Both M and N are fist shapes with the thumb tucked under the fingers.
    M tucks under THREE fingers (index+middle+ring); N tucks under TWO (index+middle).
    This rule can't directly count fingers but checks thumb tip y relative
    to the index and middle PIP joints — if the thumb is NOT below (higher y
    than) the PIP joints, it is not properly tucked.
    """

    def evaluate(
        self,
        expected: str,
        predicted: str,
        landmarks: list[float],
    ) -> Optional[LandmarkDeviation]:
        pair = {expected.upper(), predicted.upper()}
        if not ({"M", "N"} & pair):
            return None

        thumb_tip = _lm(landmarks, 4)
        index_pip = _lm(landmarks, 6)
        middle_pip = _lm(landmarks, 10)
        avg_pip_y = (index_pip[1] + middle_pip[1]) / 2

        # Thumb should be tucked UNDER the fingers → thumb tip y > avg pip y
        # (larger y = lower in the image)
        if thumb_tip[1] < avg_pip_y - 0.05:
            return LandmarkDeviation(
                rule_name="thumb_under_fingers",
                description=(
                    f"For '{expected}', the thumb must be tucked UNDER the fingers. "
                    f"Your thumb appears above or alongside the fingers. "
                    f"Tuck the thumb fully under the curled fingers. "
                    f"(M = thumb under 3 fingers; N = thumb under 2 fingers)"
                ),
                severity="error",
            )
        return None


class PalmOrientationRule:
    """
    Checks approximate palm orientation by comparing z-depth of the wrist
    vs the middle MCP. For ASL letters, the palm should typically face the
    camera: wrist.z > middle_mcp.z (wrist farther from camera, knuckles
    closer). This fires when the relative depths suggest a side-on or
    back-of-hand orientation for letters that strictly require a palm-facing
    pose (B, C, D, E, F, K, L, P, Q, V, W).
    """

    _PALM_FACING_LETTERS: set[str] = {"B", "C", "D", "E", "F", "K", "L", "P", "Q", "V", "W"}

    def evaluate(
        self,
        expected: str,
        predicted: str,
        landmarks: list[float],
    ) -> Optional[LandmarkDeviation]:
        if expected.upper() not in self._PALM_FACING_LETTERS:
            return None

        wrist = _lm(landmarks, 0)
        middle_mcp = _lm(landmarks, 9)
        z_diff = wrist[2] - middle_mcp[2]

        # z is positive pointing INTO the camera. If wrist.z ≈ middle_mcp.z
        # the hand is roughly edge-on to the camera.  If wrist.z < middle_mcp.z
        # the back of the hand faces the camera.
        # MediaPipe z values are relative so we use this as a rough signal only.
        if z_diff < -0.07:
            return LandmarkDeviation(
                rule_name="palm_orientation",
                description=(
                    f"For '{expected}', the palm should face the camera. "
                    f"Your hand appears to be turned sideways or showing the back. "
                    f"Rotate your wrist so that your palm faces directly toward the camera."
                ),
                severity="warning",
            )
        return None


# ---------------------------------------------------------------------------
# Default rule list — the order determines evaluation priority.
# More specific / higher-confidence rules come first.
# ---------------------------------------------------------------------------

def default_rules() -> list:
    """Returns a fresh list of all default rules in evaluation order."""
    return [
        FingerCrossingRule(),      # most specific (letter-pair targeted)
        ThumbUnderFingersRule(),   # most specific (letter-pair targeted)
        ThumbExtensionRule(),      # letter-specific thumb check
        FingerExtensionRule(),     # broad 4-finger extension check
        PalmOrientationRule(),     # broad orientation check (fire last)
    ]
