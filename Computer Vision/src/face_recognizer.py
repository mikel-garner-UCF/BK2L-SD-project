"""Face recognizer using dlib 128-dim embeddings + Euclidean distance.

Drop-in replacement for face_recognizer.py.
Uses the same encrypted FaceDatabase for storage.

Key differences from the ONNX/ArcFace version:
  - 128-dim embeddings (not 512)
  - Euclidean distance: lower = more similar (not cosine: higher = more similar)
  - Much better separation between different people (~0.9 gap vs ~0.003)
  - Threshold: 0.5 = match (not 0.95)

The RecognitionResult.distance field contains Euclidean distance (lower = better).
The RecognitionResult.confidence converts this to 0-1 (higher = better).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from face_embedder import FaceEmbedder
from face_database import FaceDatabase


@dataclass
class RecognitionResult:
    """Result of a face recognition attempt."""
    name: Optional[str]
    distance: float
    is_known: bool

    @property
    def confidence(self) -> float:
        """Convert Euclidean distance to 0-1 confidence.
        distance 0.0 = perfect match (100%)
        distance 0.5 = at threshold (50%)
        distance 1.0+ = definitely not a match (0%)
        """
        return max(0.0, min(1.0, 1.0 - self.distance))


class FaceRecognizer:
    """Face recognizer using dlib embeddings and Euclidean distance.

    Usage:
        recognizer = FaceRecognizer(distance_threshold=0.5)
        result = recognizer.recognize(face_crop_bgr)
        if result.is_known:
            print(f"Hello, {result.name}! (confidence: {result.confidence:.0%})")
    """

    def __init__(
        self,
        distance_threshold: float = 0.5,
        min_match_ratio: float = 0.6,
        min_margin: float = 0.05,
        db: Optional[FaceDatabase] = None,
        embedder: Optional[FaceEmbedder] = None,
        # Accept similarity_threshold for backward compatibility
        similarity_threshold: float = 0.0,
    ):
        """Create a dlib-based face recognizer.

        Args:
            distance_threshold: Maximum Euclidean distance for a template
                to count as "matching." Default 0.5. Lower = stricter.
                Same person typically: 0.2-0.4
                Different person typically: 0.8-1.2

            min_match_ratio: Minimum fraction of templates that must be
                below the threshold. Default 0.6 (60%).

            min_margin: Minimum gap between best and second-best candidate.
                Default 0.05. With dlib this is very achievable since
                the gap between same/different person is huge (~0.5).

            db: Optional FaceDatabase instance.
            embedder: Optional FaceEmbedder instance.
        """
        self._threshold = distance_threshold
        self._min_match_ratio = min_match_ratio
        self._min_margin = min_margin
        self._embedder = embedder or FaceEmbedder()
        self._db = db or FaceDatabase()

        self._enrolled: Dict[str, List[np.ndarray]] = self._db.get_all()

        print(f"[face_recognizer] Ready. {len(self._enrolled)} faculty enrolled. "
              f"Distance threshold: {self._threshold}, "
              f"Min match ratio: {self._min_match_ratio}, "
              f"Min margin: {self._min_margin}")

    @staticmethod
    def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance between two vectors."""
        return float(np.linalg.norm(a - b))

    def recognize(self, face_bgr: np.ndarray) -> RecognitionResult:
        """Recognize a face against enrolled faculty.

        Process:
        1. Compute 128-dim embedding for the input face
        2. Compare against every enrolled person using Euclidean distance
        3. Majority vote: enough templates must be below threshold
        4. Margin check: best must be clearly better than second-best

        Args:
            face_bgr: BGR face crop image.

        Returns:
            RecognitionResult with name, distance, and is_known.
        """
        if not self._enrolled:
            return RecognitionResult(name=None, distance=1.0, is_known=False)

        try:
            query = self._embedder.embed(face_bgr)
        except ValueError:
            return RecognitionResult(name=None, distance=1.0, is_known=False)

        candidates = []

        for name, templates in self._enrolled.items():
            distances = [self._euclidean_distance(query, t) for t in templates]
            avg_dist = sum(distances) / len(distances)
            min_dist = min(distances)

            # Majority vote: how many templates are below threshold?
            matches = sum(1 for d in distances if d <= self._threshold)
            match_ratio = matches / len(templates) if templates else 0.0

            print(f"[recognition] {name}: avg_dist={avg_dist:.4f}, "
                  f"min_dist={min_dist:.4f}, "
                  f"match_ratio={matches}/{len(templates)}")

            candidates.append({
                "name": name,
                "avg_dist": avg_dist,
                "min_dist": min_dist,
                "match_ratio": match_ratio,
            })

        # Sort by average distance (lowest = best match)
        candidates.sort(key=lambda c: c["avg_dist"])
        best = candidates[0]

        # Check majority vote
        if best["match_ratio"] < self._min_match_ratio:
            print(f"[recognition] → Unknown (match_ratio {best['match_ratio']:.0%} "
                  f"< {self._min_match_ratio:.0%})")
            return RecognitionResult(
                name=None,
                distance=best["avg_dist"],
                is_known=False,
            )

        # Check margin
        if len(candidates) >= 2:
            second = candidates[1]
            margin = second["avg_dist"] - best["avg_dist"]

            print(f"[recognition] Margin: {second['name']}({second['avg_dist']:.4f}) - "
                  f"{best['name']}({best['avg_dist']:.4f}) = {margin:.4f} "
                  f"(need >= {self._min_margin})")

            if margin < self._min_margin and second["match_ratio"] >= self._min_match_ratio:
                print(f"[recognition] → Unknown (margin too small)")
                return RecognitionResult(
                    name=None,
                    distance=best["avg_dist"],
                    is_known=False,
                )

        print(f"[recognition] → {best['name']} (dist={best['avg_dist']:.4f})")
        return RecognitionResult(
            name=best["name"],
            distance=best["avg_dist"],
            is_known=True,
        )

    def reload_database(self) -> None:
        """Reload enrolled faculty from the encrypted database."""
        self._enrolled = self._db.get_all()
        print(f"[face_recognizer] Reloaded. {len(self._enrolled)} faculty enrolled.")

    @property
    def enrolled_count(self) -> int:
        return len(self._enrolled)