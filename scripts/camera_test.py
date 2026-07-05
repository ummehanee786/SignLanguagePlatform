"""
camera_test.py

Task 3 - Webcam Verification

Requirements:
    - Open the webcam
    - Show the live camera feed
    - Press Q to close
    - Nothing else. No AI. No MediaPipe. No OpenCV filters.

Goal: simply verify the webcam works correctly before building
anything on top of it.
"""

import cv2


def main():
    # 0 = default system webcam. If you have multiple cameras and the
    # wrong one opens, try 1, 2, etc.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[!] Could not open webcam. Check that:")
        print("    - Your webcam is connected and not used by another app")
        print("    - No other program (Zoom, Teams, etc.) is currently using it")
        print("    - You may need to try a different camera index (1, 2, ...)")
        return

    print("Webcam opened successfully.")
    print("Press 'Q' on the video window to quit.")

    while True:
        success, frame = cap.read()

        if not success:
            print("[!] Failed to read frame from webcam. Stopping.")
            break

        cv2.imshow("Camera Test - Press Q to quit", frame)

        # waitKey(1) waits 1ms between frames and checks for a key press.
        # 0xFF masks the result so it compares cleanly across platforms.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Q pressed. Closing webcam.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()