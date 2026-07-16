"""
detector.py

This is the ONLY place in the backend that is allowed to import cv2 or
mediapipe. Everything else (services, routers, schemas) must go through
this class instead of touching those libraries directly.
"""

import cv2
import mediapipe as mp


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