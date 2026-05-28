"""
Full real-time face recognition check using dlib 128-dim embeddings.

Run this from the Computer Vision directory:
    python tests/test_face_recognition.py

Prerequisites:
  - At least one person enrolled with dlib:
    python src/enroll.py --enroll --name "Your Name"
  - All dependencies installed

Controls:
  q = quit
  s = save snapshot
  r = reload database
  +/- = adjust distance threshold (lower = stricter)
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np

from face_detection import FaceDetector
from face_tracker import FaceTracker, Track
from face_recognizer import FaceRecognizer, RecognitionResult
from face_database import FaceDatabase

# UCF colors
UCF_GOLD = (4, 201, 255)
UCF_BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (0, 0, 200)

# dlib database paths
DLIB_DB_PATH = PROJECT_ROOT / "data" / "dlib_face_embeddings.enc"
DLIB_KEY_PATH = PROJECT_ROOT / "data" / ".dlib_encryption_key"


class FPSCounter:
    def __init__(self, window_size: int = 30):
        self._window_size = window_size
        self._frame_times: list = []
        self._last_time = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self._last_time is not None:
            self._frame_times.append(now - self._last_time)
            if len(self._frame_times) > self._window_size:
                self._frame_times.pop(0)
        self._last_time = now
        if not self._frame_times:
            return 0.0
        avg = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / avg if avg > 0 else 0.0


def draw_recognition_results(
    frame: np.ndarray,
    tracks: list,
    recognition_cache: dict,
) -> np.ndarray:
    out = frame.copy()

    for track in tracks:
        box = track.bbox
        result = recognition_cache.get(track.track_id)

        if result and result.is_known:
            color = GREEN
            label = f"{result.name} ({result.confidence:.0%})"
        elif result and not result.is_known:
            color = UCF_GOLD
            label = "Unknown"
        else:
            color = UCF_GOLD
            label = f"#{track.track_id}"

        cv2.rectangle(out, (box.x, box.y), (box.x2, box.y2), color, 2)

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = box.x
        text_y = max(0, box.y - 10)

        cv2.rectangle(
            out,
            (text_x, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            color, -1,
        )
        cv2.putText(out, label, (text_x + 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, UCF_BLACK, 2)

        if result and result.is_known:
            bar_width = int(box.width * result.confidence)
            bar_y = box.y2 + 5
            cv2.rectangle(out, (box.x, bar_y), (box.x + bar_width, bar_y + 8),
                          GREEN, -1)
            cv2.rectangle(out, (box.x, bar_y), (box.x + box.width, bar_y + 8),
                          color, 1)

    return out


def main() -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return 1

    print("=== Knightro Face Recognition Test (dlib) ===")
    print()

    detector = FaceDetector(min_confidence=0.5)
    tracker = FaceTracker(iou_threshold=0.3, max_missed_frames=15)

    # Load dlib database
    db = FaceDatabase(db_path=DLIB_DB_PATH, key_path=DLIB_KEY_PATH)

    recognizer = FaceRecognizer(distance_threshold=0.5, db=db)

    if recognizer.enrolled_count == 0:
        print("WARNING: No faculty enrolled in dlib database!")
        print("Enroll someone first: python src/enroll.py --enroll --name \"Your Name\"")
        print()

    print("Controls:")
    print("  q = quit")
    print("  s = save snapshot")
    print("  r = reload database")
    print("  +/- = adjust distance threshold (lower = stricter)")
    print()

    fps_counter = FPSCounter(window_size=30)
    recognition_cache: dict = {}
    recognized_ids: set = set()

    with detector:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)

            detections = detector.detect(frame)
            active_tracks = tracker.update(detections)

            current_track_ids = {t.track_id for t in active_tracks}

            stale_ids = set(recognition_cache.keys()) - current_track_ids
            for stale_id in stale_ids:
                del recognition_cache[stale_id]
                recognized_ids.discard(stale_id)

            for track in active_tracks:
                if (track.track_id not in recognized_ids
                        and track.frames_seen >= 3):
                    box = track.bbox
                    margin = 20
                    y1 = max(0, box.y - margin)
                    y2 = min(frame.shape[0], box.y2 + margin)
                    x1 = max(0, box.x - margin)
                    x2 = min(frame.shape[1], box.x2 + margin)
                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size > 0:
                        result = recognizer.recognize(face_crop)
                        recognition_cache[track.track_id] = result
                        recognized_ids.add(track.track_id)

                        if result.is_known:
                            print(f"  Recognized: {result.name} "
                                  f"(distance: {result.distance:.3f}, "
                                  f"confidence: {result.confidence:.0%})")
                        else:
                            print(f"  Unknown face (closest distance: "
                                  f"{result.distance:.3f})")

            display = draw_recognition_results(frame, active_tracks,
                                               recognition_cache)

            fps = fps_counter.tick()
            threshold = recognizer._threshold
            status = (f"FPS: {fps:5.1f}  |  "
                      f"Tracks: {tracker.active_track_count}  |  "
                      f"Enrolled: {recognizer.enrolled_count}  |  "
                      f"Threshold: {threshold:.2f}")

            for color, thickness in [(UCF_BLACK, 4), (UCF_BLACK, 2)]:
                cv2.putText(display, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness)

            cv2.imshow("Knightro Face Recognition (dlib)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                path = PROJECT_ROOT / "data" / f"recognition_{int(time.time())}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), display)
                print(f"Saved snapshot to {path}")
            elif key == ord('r'):
                recognizer.reload_database()
                recognition_cache.clear()
                recognized_ids.clear()
                print("Database reloaded and cache cleared.")
            elif key == ord('+') or key == ord('='):
                recognizer._threshold = min(1.0, recognizer._threshold + 0.05)
                print(f"Threshold: {recognizer._threshold:.2f}")
            elif key == ord('-'):
                recognizer._threshold = max(0.05, recognizer._threshold - 0.05)
                print(f"Threshold: {recognizer._threshold:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())