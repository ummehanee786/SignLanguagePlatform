"""
webcam_demo.py

Demo/display script for the reusable Camera module.

This file only handles displaying frames on screen and quitting on 'Q'.
All actual camera logic (opening, reading, releasing, error handling)
lives in camera.py - kept separate on purpose, per Weekend Task 2's
"don't combine everything into one script" instruction.
"""

import sys
from pathlib import Path

import cv2

# Make sure Python can find camera.py regardless of where this script
# is run from (no hardcoded absolute paths).
sys.path.append(str(Path(__file__).resolve().parent))

from camera import Camera, CameraError


def main():
    try:
        camera = Camera(camera_index=0)
    except CameraError as e:
        print(f"[!] {e}")
        return

    print("Webcam opened successfully.")
    print("Press 'Q' on the video window to quit.")

    window_name = "Webcam Demo - Press Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 100, 100)
    cv2.resizeWindow(window_name, 640, 480)

    with camera:
        while True:
            frame = camera.get_frame()

            if frame is None:
                print("[!] Failed to read frame from webcam. Stopping.")
                break

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Q pressed. Closing webcam.")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()