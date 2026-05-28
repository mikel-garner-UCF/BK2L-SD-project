"""Face embedder using dlib (128-dimensional embeddings).

Drop-in replacement for face_embedder.py.
Uses the face_recognition library (built on dlib) to compute
128-dim embeddings using Euclidean distance instead of cosine similarity.

Installation:
    conda install -c conda-forge dlib face_recognition
"""

import numpy as np
import cv2


class FaceEmbedder:
    """Compute 128-dim face embeddings using dlib.

    Usage:
        embedder = FaceEmbedder()
        embedding = embedder.embed(face_crop_bgr)
        # embedding is a 128-dim numpy array
    """

    def __init__(self):
        try:
            import face_recognition
            self._fr = face_recognition
        except ImportError:
            raise ImportError(
                "face_recognition not installed. "
                "Run: conda install -c conda-forge dlib face_recognition"
            )
        print(f"[face_embedder] dlib model loaded. Output dim: 128")

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """Compute a 128-dim embedding from a BGR face crop.

        Args:
            face_bgr: BGR face crop image (OpenCV format).

        Returns:
            128-dim numpy float32 array.

        Raises:
            ValueError: If no face could be encoded.
        """
        # Ensure uint8
        if face_bgr.dtype != np.uint8:
            face_bgr = (face_bgr * 255).astype(np.uint8) if face_bgr.max() <= 1.0 else face_bgr.astype(np.uint8)

        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        # Make sure it's 3 channels (strip alpha if present)
        if len(face_rgb.shape) == 2:
            face_rgb = cv2.cvtColor(face_rgb, cv2.COLOR_GRAY2RGB)
        elif face_rgb.shape[2] == 4:
            face_rgb = face_rgb[:, :, :3]

        # Make contiguous in memory (dlib requires this)
        face_rgb = np.ascontiguousarray(face_rgb)

        # Tell face_recognition the entire image IS the face
        h, w = face_rgb.shape[:2]
        face_location = [(0, w, h, 0)]  # (top, right, bottom, left)

        encodings = self._fr.face_encodings(
            face_rgb,
            known_face_locations=face_location,
            num_jitters=1,
            model="large",
        )

        if not encodings:
            raise ValueError("Could not compute face encoding")

        return encodings[0].astype(np.float32)