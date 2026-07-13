"""
detector.py

This is the ONLY place in the backend that is allowed to import cv2 or
mediapipe. Everything else (services, routers, schemas) must go through
this class instead of touching those libraries directly.

This keeps the AI/computer-vision logic completely separate from the
API layer - the FastAPI app doesn't need to know or care HOW hand
detection works internally, only that it can call extract_landmarks()
and get a result back. This is what makes it possible to swap in the
real trained classifier later (Week 3) without rewriting the API.
"""

import cv2
import mediapipe as mp


class HandLandmarkDetector:
    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.3,
    ):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
        )

    def extract_landmarks(self, frame):
        """
        Given a BGR image/frame (as read by cv2.imread or a webcam
        frame), returns a flat list of 63 values (x, y, z for each of
        the 21 landmarks) for the first detected hand, or None if no
        hand was detected.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]

        values = []
        for lm in hand_landmarks.landmark:
            values.extend([lm.x, lm.y, lm.z])

        return values

    def close(self):
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()