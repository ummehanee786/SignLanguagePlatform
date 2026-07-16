"""
hand_tracking.py

Weekend Task 3 - MediaPipe Hand Tracking
Weekend Task 4 - FPS Counter
Weekend Task 5 - Landmark Extraction
Weekend Task 6 - Save Landmark Data
Bonus - Thumb-Index Euclidean Distance

Refactored to reuse HandLandmarkDetector instead of setting up
MediaPipe directly - this is now the only script duplicating the AI
module's logic, and it doesn't duplicate anymore: all MediaPipe
setup/calls go through detector.py.

Controls while the window is focused:
    Q  -  quit
    P  -  print current landmark coordinates to the terminal
    S  -  save current landmark coordinates to a JSON file in captures/
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# Reach camera.py (in app/utils/) and detector.py (in this same folder),
# without hardcoding an absolute path.
utils_dir = Path(__file__).resolve().parent.parent.parent / "utils"
sys.path.append(str(utils_dir))
sys.path.append(str(Path(__file__).resolve().parent))

from camera import Camera, CameraError
from detector import HandLandmarkDetector


def landmarks_to_dicts(hand_landmarks):
    """Task 5: pull (x, y, z) out of a MediaPipe result as plain dicts,
    for printing/saving/distance calculations."""
    return [
        {"id": idx, "x": lm.x, "y": lm.y, "z": lm.z}
        for idx, lm in enumerate(hand_landmarks.landmark)
    ]


def print_landmarks(all_hands_landmarks):
    """Task 5: print landmark coordinates in a structured, readable format."""
    print("\n" + "=" * 50)
    for hand_index, landmarks in enumerate(all_hands_landmarks):
        print(f"Hand {hand_index + 1}")
        for point in landmarks:
            print(f"  Landmark {point['id']:>2} : "
                  f"x={point['x']:.4f}  y={point['y']:.4f}  z={point['z']:.4f}")
    print("=" * 50)


def save_landmarks(all_hands_landmarks, captures_dir: Path):
    """Task 6: save the current frame's landmark data to a JSON file."""
    captures_dir.mkdir(parents=True, exist_ok=True)

    existing = list(captures_dir.glob("capture_*.json"))
    next_number = len(existing) + 1
    filename = captures_dir / f"capture_{next_number:03d}.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "num_hands": len(all_hands_landmarks),
        "hands": [
            {"hand_number": i + 1, "landmarks": landmarks}
            for i, landmarks in enumerate(all_hands_landmarks)
        ],
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[i] Saved landmark data to: {filename}")


def euclidean_distance(p1, p2):
    """Bonus: straight-line distance between two landmark points (2D)."""
    return ((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2) ** 0.5


def main():
    try:
        camera = Camera(camera_index=0)
    except CameraError as e:
        print(f"[!] {e}")
        return

    detector = HandLandmarkDetector(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # captures/ lives at the project root, next to backend/, datasets/, etc.
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    captures_dir = project_root / "captures"

    print("Webcam + HandLandmarkDetector started.")
    print("Controls: Q = quit | P = print landmarks | S = save landmarks")

    window_name = "Hand Tracking - Press Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 100, 100)
    cv2.resizeWindow(window_name, 640, 480)

    prev_frame_time = 0.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (0, 255, 0)
    warning_color = (0, 0, 255)
    distance_color = (255, 200, 0)
    thickness = 2

    THUMB_TIP_ID = 4
    INDEX_TIP_ID = 8

    with camera:
        while True:
            frame = camera.get_frame()

            if frame is None:
                print("[!] Failed to read frame from webcam. Stopping.")
                break

            current_frame_time = time.time()
            time_diff = current_frame_time - prev_frame_time
            fps = 1.0 / time_diff if time_diff > 0 else 0.0
            prev_frame_time = current_frame_time

            multi_hand_landmarks = detector.process(frame)

            num_hands = 0
            all_hands_landmarks = []

            if multi_hand_landmarks:
                num_hands = len(multi_hand_landmarks)
                for hand_landmarks in multi_hand_landmarks:
                    detector.draw_landmarks(frame, hand_landmarks)
                    all_hands_landmarks.append(landmarks_to_dicts(hand_landmarks))

            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                        font, 1, text_color, thickness)

            if num_hands > 0:
                cv2.putText(frame, f"Hands detected: {num_hands}", (10, 65),
                            font, 1, text_color, thickness)
            else:
                cv2.putText(frame, "No Hand Detected", (10, 65),
                            font, 1, warning_color, thickness)

            for i, landmarks in enumerate(all_hands_landmarks):
                thumb_tip = landmarks[THUMB_TIP_ID]
                index_tip = landmarks[INDEX_TIP_ID]
                distance = euclidean_distance(thumb_tip, index_tip)
                cv2.putText(
                    frame,
                    f"Hand {i + 1} pinch distance: {distance:.3f}",
                    (10, 100 + (i * 35)),
                    font, 0.8, distance_color, thickness,
                )

            cv2.putText(frame, "P: print  S: save  Q: quit", (10, frame.shape[0] - 15),
                        font, 0.6, (200, 200, 200), 1)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Q pressed. Closing.")
                break
            elif key == ord("p"):
                if all_hands_landmarks:
                    print_landmarks(all_hands_landmarks)
                else:
                    print("[i] 'P' pressed but no hand currently detected.")
            elif key == ord("s"):
                if all_hands_landmarks:
                    save_landmarks(all_hands_landmarks, captures_dir)
                else:
                    print("[i] 'S' pressed but no hand currently detected - nothing saved.")

    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()