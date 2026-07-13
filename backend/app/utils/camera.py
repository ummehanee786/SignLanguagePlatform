"""
camera.py

Reusable webcam module.

This is intentionally separate from any script that *uses* the camera
(e.g. a demo/display script). Keeping the camera logic here means it
can be imported and reused later by the hand tracking module, the
gesture recognition module, or anything else - without duplicating
webcam-handling code everywhere.
"""

import cv2


class CameraError(Exception):
    """Raised when the camera cannot be opened or read from."""
    pass


class Camera:
    """
    Thin wrapper around cv2.VideoCapture that handles initialization
    failures gracefully instead of crashing, and exposes a simple
    get_frame() / release() interface.
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise CameraError(
                f"Could not open camera at index {camera_index}. "
                f"Check that a webcam is connected and not already in "
                f"use by another application, or try a different index."
            )

    def get_frame(self):
        """
        Read a single frame from the camera.

        Returns:
            The captured frame (numpy array), or None if the read failed
            (e.g. camera was disconnected mid-stream).
        """
        success, frame = self.cap.read()

        if not success:
            return None

        return frame

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def release(self):
        """Release the camera. Safe to call even if already released."""
        if self.cap is not None:
            self.cap.release()

    # Allows using this class with a "with" statement:
    #   with Camera() as cam:
    #       frame = cam.get_frame()
    # so release() is always called automatically, even if an error
    # happens inside the block.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()