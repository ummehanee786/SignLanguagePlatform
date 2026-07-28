"""
detector.py

This is the ONLY place in the backend that is allowed to import cv2 or
mediapipe. Everything else (services, routers, schemas) must go through
this class instead of touching those libraries directly.
"""

from typing import Optional

import cv2
import numpy as np
import mediapipe as mp


def decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decodes raw image bytes (e.g. straight from an UploadFile in a
    FastAPI route) into a BGR frame, the same shape/format
    HandLandmarkDetector.extract_landmarks() expects.

    This exists so that nothing outside this file ever needs to import
    cv2 just to turn an uploaded file's bytes into a usable frame -
    api/predict.py and gesture_service.py stay cv2-free and only pass
    raw bytes around.

    Returns None if the bytes couldn't be decoded as an image (corrupt
    upload, wrong file type, etc.) instead of raising - callers should
    treat None the same way they'd treat "no hand detected".
    """
    if not image_bytes:
        return None
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return frame  # None if decoding failed


class HandLandmarkDetector:
    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self._hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame):
        """
        Returns the raw list of detected hands' landmark objects (or
        None), for callers that need to draw on the frame or do their
        own custom extraction - e.g. the live webcam demo.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)
        return results.multi_hand_landmarks

    def extract_landmarks(self, frame):
        """
        Given a BGR frame, returns a flat list of 63 values (x, y, z
        for each of the 21 landmarks) for the first detected hand, or
        None if no hand was detected. Used by preprocessing.
        """
        multi_hand_landmarks = self.process(frame)
        if not multi_hand_landmarks:
            return None

        hand_landmarks = multi_hand_landmarks[0]
        values = []
        for lm in hand_landmarks.landmark:
            values.extend([lm.x, lm.y, lm.z])
        return values

    def extract_landmarks_with_metadata(self, frame) -> dict:
        """
        Same underlying detection as extract_landmarks(), but also
        reports how many hands were actually seen in the frame - used
        by the inference engine to explicitly reject multi-hand frames
        instead of silently picking one hand and discarding the rest.

        Note: for this to actually be able to *see* a second hand, the
        detector instance calling this method needs to have been
        constructed with max_num_hands >= 2 - extract_landmarks() (used
        for single-hand dataset building) intentionally uses
        max_num_hands=1 and would never observe a second hand even if
        one were present, which is correct for that use case but wrong
        for this one.

        Returns:
            {
                "hand_count": int,
                "landmarks": list[float] | None,  # first hand's 63 values, if any
            }
        """
        multi_hand_landmarks = self.process(frame)
        hand_count = len(multi_hand_landmarks) if multi_hand_landmarks else 0

        landmarks = None
        if hand_count >= 1:
            values = []
            for lm in multi_hand_landmarks[0].landmark:
                values.extend([lm.x, lm.y, lm.z])
            landmarks = values

        return {"hand_count": hand_count, "landmarks": landmarks}

    def draw_landmarks(self, frame, hand_landmarks):
        """Draws the 21 landmarks + connections onto frame, in place."""
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style(),
        )

    def close(self):
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()