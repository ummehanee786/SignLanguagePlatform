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
        self.mp_pose = mp.solutions.pose
        self._pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
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
        reports how many hands were actually seen in the frame and
        pose/visibility metadata.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_results = self._hands.process(rgb_frame)
        pose_results = self._pose.process(rgb_frame)

        multi_hand_landmarks = hands_results.multi_hand_landmarks
        hand_count = len(multi_hand_landmarks) if multi_hand_landmarks else 0

        landmarks = None
        if hand_count >= 1:
            values = []
            for lm in multi_hand_landmarks[0].landmark:
                values.extend([lm.x, lm.y, lm.z])
            landmarks = values

        has_person = pose_results.pose_landmarks is not None
        upper_body_visible = True
        
        # Check upper body visibility
        # Nose(0), Left shoulder(11), Right shoulder(12), Left elbow(13), Right elbow(14)
        if has_person:
            landmarks_pose = pose_results.pose_landmarks.landmark
            upper_body_indices = [0, 11, 12, 13, 14]
            # Ensure none are under a threshold (e.g. 0.5)
            for idx in upper_body_indices:
                if idx < len(landmarks_pose) and landmarks_pose[idx].visibility < 0.5:
                    upper_body_visible = False
                    break
        else:
            upper_body_visible = False

        partial_hand_visible = False
        if hand_count > 0:
            for hand_lms in multi_hand_landmarks:
                for lm in hand_lms.landmark:
                    # If very close to boundaries, flag as partial hand visibility
                    if lm.x < 0.01 or lm.x > 0.99 or lm.y < 0.01 or lm.y > 0.99:
                        partial_hand_visible = True
                        break

        hand_centered = True
        if hand_count > 0:
            # Check if hand centroid / wrist is reasonably close to center (0.5, 0.5)
            # Center tolerance of 0.25 on each side
            wrist = multi_hand_landmarks[0].landmark[0]
            if abs(wrist.x - 0.5) > 0.25 or abs(wrist.y - 0.5) > 0.25:
                hand_centered = False

        return {
            "hand_count": hand_count,
            "landmarks": landmarks,
            "has_person": has_person,
            "upper_body_visible": upper_body_visible,
            "partial_hand_visible": partial_hand_visible,
            "hand_centered": hand_centered,
        }

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
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()