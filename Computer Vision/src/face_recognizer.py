
"""
Face recognizer using DUAL metrics: Cosine Similarity + Euclidean Distance.

This is an improved version of face_recognizer.py that uses BOTH distance
metrics together for more robust recognition. Using two metrics gives better
separation between enrolled people because:

  - Cosine similarity measures the ANGLE between embeddings
    (good at: "do these faces point in the same direction?")

  - Euclidean distance measures the actual DISTANCE in embedding space
    (good at: "how far apart are these faces?")

Two faces can have very similar cosine scores (0.983 vs 0.980) but
noticeably different Euclidean distances (0.184 vs 0.200). By combining
both, we get a composite score with better discrimination.

The math for L2-normalized vectors:
    euclidean = sqrt(2 - 2 * cosine_similarity)
    
    So cosine 0.983 → euclidean 0.184
       cosine 0.980 → euclidean 0.200
       cosine 0.950 → euclidean 0.316

The gap in cosine space (0.003) becomes a larger gap in euclidean space
(0.016), and the composite score amplifies this further.

Drop-in replacement for face_recognizer.py — same API, better separation.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from face_embedder import FaceEmbedder
from face_database import FaceDatabase


##################### Recognition result #####################

@dataclass
class RecognitionResult:
    """The result of trying to recognize a single face.

    Attributes:
        name: The matched person's name, or None if unknown.
        distance: Composite similarity score (higher = more similar).
        is_known: True if the face matched someone in the database.
    """
    name: Optional[str]
    distance: float
    is_known: bool

    @property
    def confidence(self) -> float:
        return max(0.0, min(1.0, self.distance))


###################### The recognizer #####################

class FaceRecognizer:
    """Face recognizer using dual metrics (cosine + euclidean).

    Usage:
        recognizer = FaceRecognizer()
        result = recognizer.recognize(face_crop)
        if result.is_known:
            print(f"Hello, {result.name}!")

    The composite score is calculated as:
        composite = (cosine_weight * cosine_sim) + (euclidean_weight * (1 - normalized_euclidean))

    This gives a single score where higher = better match, but with
    better separation than cosine alone.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.45,
        min_match_ratio: float = 0.6,
        cosine_weight: float = 0.4,
        euclidean_weight: float = 0.6,
        min_margin: float = 0.005,
        db: Optional[FaceDatabase] = None,
        embedder: Optional[FaceEmbedder] = None,
    ):
        """Create a face recognizer with dual metrics.

        Args:
            similarity_threshold: Minimum cosine similarity for a single
                template to count as "matching" in the majority vote.

            min_match_ratio: Minimum fraction of templates that must match.
                Default 0.6 = at least 60% of templates must be above threshold.

            cosine_weight: Weight for cosine similarity in composite score.
            euclidean_weight: Weight for euclidean component in composite score.
                These should sum to 1.0. Higher euclidean_weight gives more
                emphasis to absolute distance (better separation).

            min_margin: Minimum composite score gap between #1 and #2
                candidate. Default 0.005 — higher than cosine-only (0.003)
                because the composite score has a wider range.

            db: Optional FaceDatabase instance.
            embedder: Optional FaceEmbedder instance.
        """
        self._threshold = similarity_threshold
        self._min_match_ratio = min_match_ratio
        self._cosine_weight = cosine_weight
        self._euclidean_weight = euclidean_weight
        self._min_margin = min_margin
        self._embedder = embedder or FaceEmbedder()
        self._db = db or FaceDatabase()

        self._enrolled: Dict[str, list] = self._db.get_all()
        print(f"[face_recognizer] Ready. {len(self._enrolled)} faculty enrolled. "
              f"Similarity threshold: {self._threshold}, "
              f"Min match ratio: {self._min_match_ratio}, "
              f"Weights: cosine={self._cosine_weight}, euclidean={self._euclidean_weight}")

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized vectors. Range: -1 to 1."""
        return float(np.dot(a, b))

    @staticmethod
    def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance between two vectors. Range: 0 to ~2 for unit vectors."""
        return float(np.linalg.norm(a - b))

    def _composite_score(self, query: np.ndarray, template: np.ndarray) -> dict:
        """Compute both metrics and a composite score for one template.

        Returns dict with cosine_sim, euclidean_dist, and composite score.
        """
        cosine = self._cosine_similarity(query, template)
        euclidean = self._euclidean_distance(query, template)

        # Normalize euclidean to 0-1 range (for L2-normalized vectors, max is 2.0)
        # Then invert so higher = more similar (matching cosine convention)
        euclidean_normalized = 1.0 - (euclidean / 2.0)

        # Weighted composite
        composite = (self._cosine_weight * cosine) + (self._euclidean_weight * euclidean_normalized)

        return {
            "cosine": cosine,
            "euclidean": euclidean,
            "euclidean_norm": euclidean_normalized,
            "composite": composite,
        }

    def recognize(self, face_bgr: np.ndarray) -> RecognitionResult:
        """Recognize a face using dual metrics.

        Process:
        1. Compute embeddings and both distance metrics for every enrolled person
        2. Rank by composite score (cosine + euclidean combined)
        3. Majority vote check (enough templates must match)
        4. Margin check (best must be clearly better than second-best)

        Args:
            face_bgr: BGR face crop image.

        Returns:
            RecognitionResult with the match or unknown.
        """
        if not self._enrolled:
            return RecognitionResult(name=None, distance=0.0, is_known=False)

        query_embedding = self._embedder.embed(face_bgr)

        # Step 1: Score every enrolled person with both metrics
        candidates = []

        for name, templates in self._enrolled.items():
            scores = [self._composite_score(query_embedding, t) for t in templates]

            cosine_sims = [s["cosine"] for s in scores]
            euclidean_dists = [s["euclidean"] for s in scores]
            composites = [s["composite"] for s in scores]

            # Majority vote uses cosine threshold (same as original)
            matches = sum(1 for c in cosine_sims if c >= self._threshold)
            match_ratio = matches / len(templates) if templates else 0.0

            avg_cosine = sum(cosine_sims) / len(cosine_sims)
            avg_euclidean = sum(euclidean_dists) / len(euclidean_dists)
            avg_composite = sum(composites) / len(composites)

            print(f"[recognition] {name}: "
                  f"cosine={avg_cosine:.4f}, "
                  f"euclidean={avg_euclidean:.4f}, "
                  f"composite={avg_composite:.4f}, "
                  f"match_ratio={matches}/{len(templates)}")

            candidates.append({
                "name": name,
                "avg_cosine": avg_cosine,
                "avg_euclidean": avg_euclidean,
                "avg_composite": avg_composite,
                "match_ratio": match_ratio,
            })

        # Step 2: Sort by COMPOSITE score (not just cosine)
        candidates.sort(key=lambda c: c["avg_composite"], reverse=True)

        best = candidates[0]

        # Step 3: Majority vote
        if best["match_ratio"] < self._min_match_ratio:
            print(f"[recognition] → Unknown (match_ratio {best['match_ratio']:.0%} "
                  f"< {self._min_match_ratio:.0%})")
            return RecognitionResult(
                name=None,
                distance=best["avg_composite"],
                is_known=False,
            )

        # Step 4: Margin check using composite score
        if len(candidates) >= 2:
            second = candidates[1]
            margin = best["avg_composite"] - second["avg_composite"]

            print(f"[recognition] Margin: {best['name']}({best['avg_composite']:.4f}) - "
                  f"{second['name']}({second['avg_composite']:.4f}) = {margin:.4f} "
                  f"(need >= {self._min_margin})")

            if margin < self._min_margin and second["match_ratio"] >= self._min_match_ratio:
                print(f"[recognition] → Unknown (margin too small)")
                return RecognitionResult(
                    name=None,
                    distance=best["avg_composite"],
                    is_known=False,
                )

        print(f"[recognition] → {best['name']} "
              f"(composite={best['avg_composite']:.4f}, "
              f"cosine={best['avg_cosine']:.4f}, "
              f"euclidean={best['avg_euclidean']:.4f})")

        return RecognitionResult(
            name=best["name"],
            distance=best["avg_composite"],
            is_known=True,
        )

    def reload_database(self) -> None:
        """Reload enrolled faculty from the encrypted database."""
        self._db = FaceDatabase()
        self._enrolled = self._db.get_all()
        print(f"[face_recognizer] Reloaded. {len(self._enrolled)} faculty enrolled.")

    @property
    def enrolled_count(self) -> int:
        return len(self._enrolled)

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value
        print(f"[face_recognizer] Threshold updated to {value}")