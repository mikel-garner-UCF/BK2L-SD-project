"""Knightro Faculty Enrollment Tool — dlib version.

Uses dlib 128-dim embeddings stored in an encrypted database
(separate from the old ONNX database).

Database files:
  - Computer Vision/data/dlib_face_embeddings.enc  (encrypted embeddings)
  - Computer Vision/data/.dlib_encryption_key      (encryption key)

The old ONNX database is left untouched as a backup:
  - Computer Vision/data/face_embeddings.enc
  - Computer Vision/data/.encryption_key

Usage:
    python src/enroll.py --enroll --name "Dr. Smith"
    python src/enroll.py --list
    python src/enroll.py --remove --name "Dr. Smith"
    python src/enroll.py --test
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np

from face_detection import FaceDetector
from face_embedder import FaceEmbedder
from face_database import FaceDatabase

# dlib database paths (separate from ONNX)
DLIB_DB_PATH = PROJECT_ROOT / "data" / "dlib_face_embeddings.enc"
DLIB_KEY_PATH = PROJECT_ROOT / "data" / ".dlib_encryption_key"


def get_db() -> FaceDatabase:
    """Get the dlib face database."""
    return FaceDatabase(db_path=DLIB_DB_PATH, key_path=DLIB_KEY_PATH)


def enroll_faculty(name: str, photo_path: str | None = None) -> bool:
    """Enroll a faculty member using dlib embeddings."""
    print(f"\n=== Enrolling: {name} (dlib 128-dim) ===")

    detector = FaceDetector(min_confidence=0.5)
    embedder = FaceEmbedder()

    if photo_path:
        crops = _capture_from_photo(detector, photo_path)
    else:
        crops = _capture_from_webcam(detector, num_captures=10)

    if not crops:
        print("ERROR: No faces captured.")
        return False

    print(f"\nComputing dlib embeddings from {len(crops)} captures...")
    embeddings = []
    for i, crop in enumerate(crops):
        try:
            emb = embedder.embed(crop)
            embeddings.append(emb)
            print(f"  Embedding {i+1}/{len(crops)}: dim={len(emb)}")
        except ValueError as e:
            print(f"  Skipped capture {i+1}: {e}")

    if not embeddings:
        print("ERROR: No embeddings computed.")
        return False

    # Quality check: internal consistency
    if len(embeddings) >= 2:
        pairwise = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                pairwise.append(np.linalg.norm(embeddings[i] - embeddings[j]))
        avg_self = np.mean(pairwise)
        print(f"\n  Internal consistency: avg self-distance = {avg_self:.4f}")
        if avg_self > 0.45:
            print(f"  WARNING: Self-distance is high. Some captures may be bad.")

    # Quality check: separation from existing people
    db = get_db()
    existing = db.get_all()
    if existing:
        print(f"\n  Separation check against {len(existing)} enrolled:")
        new_centroid = np.mean(embeddings, axis=0)
        for other_name, other_templates in existing.items():
            if other_name == name:
                continue
            other_centroid = np.mean(other_templates, axis=0)
            dist = np.linalg.norm(new_centroid - other_centroid)
            status = "GOOD" if dist > 0.6 else "OK" if dist > 0.4 else "CLOSE"
            print(f"    vs {other_name}: distance = {dist:.4f} [{status}]")

    # Save
    db.add_face(name, embeddings)
    db.save()
    print(f"\nSUCCESS: {name} enrolled with {len(embeddings)} dlib templates.")
    return True


def _capture_from_webcam(detector: FaceDetector, num_captures: int = 10) -> list:
    """Capture face crops from webcam with angle guidance."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return []

    print(f"\nCapturing {num_captures} face images.")
    print("Vary your head angle for better recognition:")
    print("  Captures 1-3: Look STRAIGHT")
    print("  Captures 4-5: Turn slightly LEFT")
    print("  Captures 6-7: Turn slightly RIGHT")
    print("  Captures 8-9: Tilt slightly UP/DOWN")
    print("  Capture 10:   45-degree angle")
    print()
    print("  SPACE = capture | 'a' = auto-capture | 'q' = quit")
    print()

    guidance = [
        "Look STRAIGHT at camera", "Look STRAIGHT at camera", "Look STRAIGHT at camera",
        "Turn head slightly LEFT", "Turn head slightly LEFT",
        "Turn head slightly RIGHT", "Turn head slightly RIGHT",
        "Tilt head slightly UP", "Tilt head slightly DOWN",
        "Try a 45-degree angle",
    ]

    crops = []
    auto_mode = False
    last_capture = 0.0

    with detector:
        while len(crops) < num_captures:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            boxes = detector.detect(frame)
            display = frame.copy()

            # Draw face boxes
            for box in boxes:
                cv2.rectangle(display, (box.x, box.y), (box.x2, box.y2),
                              (0, 199, 255), 2)

            # Status
            status = f"Captured: {len(crops)}/{num_captures}"
            if auto_mode:
                status += " [AUTO]"
            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 199, 255), 2)

            # Guidance
            if len(crops) < len(guidance):
                cv2.putText(display, guidance[len(crops)], (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

            cv2.imshow("Knightro Enrollment (dlib)", display)

            key = cv2.waitKey(1) & 0xFF
            should_capture = False

            if key == ord('q'):
                break
            elif key == ord(' '):
                should_capture = True
            elif key == ord('a'):
                auto_mode = not auto_mode
                print(f"Auto-capture {'ON' if auto_mode else 'OFF'}")

            if auto_mode and (time.time() - last_capture) >= 0.5:
                should_capture = True

            if should_capture and len(boxes) == 1:
                box = boxes[0]
                margin = 20
                y1 = max(0, box.y - margin)
                y2 = min(frame.shape[0], box.y2 + margin)
                x1 = max(0, box.x - margin)
                x2 = min(frame.shape[1], box.x2 + margin)
                crop = frame[y1:y2, x1:x2].copy()

                if crop.size > 0:
                    crops.append(crop)
                    last_capture = time.time()
                    print(f"  Captured {len(crops)}/{num_captures}")

                    # Flash green
                    cv2.rectangle(display, (0, 0),
                                  (display.shape[1], display.shape[0]),
                                  (0, 255, 0), 10)
                    cv2.imshow("Knightro Enrollment (dlib)", display)
                    cv2.waitKey(200)

    cap.release()
    cv2.destroyAllWindows()
    return crops


def _capture_from_photo(detector: FaceDetector, photo_path: str) -> list:
    """Extract face from a photo."""
    image = cv2.imread(photo_path)
    if image is None:
        print(f"ERROR: Could not read {photo_path}")
        return []

    with detector:
        boxes = detector.detect(image)

    if not boxes:
        print("ERROR: No face detected in photo.")
        return []

    box = max(boxes, key=lambda b: b.width * b.height)
    margin = 20
    y1 = max(0, box.y - margin)
    y2 = min(image.shape[0], box.y2 + margin)
    x1 = max(0, box.x - margin)
    x2 = min(image.shape[1], box.x2 + margin)
    return [image[y1:y2, x1:x2].copy()]


def test_separation():
    """Test Euclidean distance separation between all enrolled people."""
    db = get_db()
    enrolled = db.get_all()

    if len(enrolled) < 2:
        print("Need at least 2 enrolled people to test separation.")
        return

    names = sorted(enrolled.keys())
    print(f"\n=== dlib Separation Test ({len(names)} people) ===")
    print(f"Threshold: 0.5 (below = match, above = different person)\n")

    print("Cross-person distances (higher = better):")
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i >= j:
                continue
            dists = [
                np.linalg.norm(e1 - e2)
                for e1 in enrolled[a]
                for e2 in enrolled[b]
            ]
            avg = np.mean(dists)
            mn = np.min(dists)
            status = "GOOD" if avg > 0.6 else "OK" if avg > 0.4 else "CLOSE"
            print(f"  {a} vs {b}: avg={avg:.4f}, min={mn:.4f} [{status}]")

    print("\nSelf-similarity (lower = more consistent):")
    for name, templates in sorted(enrolled.items()):
        if len(templates) < 2:
            print(f"  {name}: only 1 template")
            continue
        dists = [
            np.linalg.norm(templates[i] - templates[j])
            for i in range(len(templates))
            for j in range(i + 1, len(templates))
        ]
        print(f"  {name}: avg={np.mean(dists):.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Knightro Face Enrollment (dlib)",
    )

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--enroll", action="store_true")
    action.add_argument("--remove", action="store_true")
    action.add_argument("--list", action="store_true")
    action.add_argument("--test", action="store_true")

    parser.add_argument("--name", type=str)
    parser.add_argument("--photo", type=str, default=None)

    args = parser.parse_args()

    if args.enroll:
        if not args.name:
            print("ERROR: --name required")
            return 1
        return 0 if enroll_faculty(args.name, args.photo) else 1

    elif args.remove:
        if not args.name:
            print("ERROR: --name required")
            return 1
        db = get_db()
        if db.remove_face(args.name):
            db.save()
            print(f"Removed {args.name}")
        else:
            print(f"'{args.name}' not found")
        return 0

    elif args.list:
        db = get_db()
        names = db.list_enrolled()
        if names:
            print(f"\nEnrolled faculty ({len(names)}) — dlib database:")
            for n in names:
                templates = db.get_embedding(n)
                count = len(templates) if templates else 0
                print(f"  - {n} ({count} templates, 128-dim)")
        else:
            print("No faculty enrolled in dlib database.")
        return 0

    elif args.test:
        test_separation()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())