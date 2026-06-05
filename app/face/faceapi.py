"""
Face detection + embedding using InsightFace (SCRFD detector + ArcFace recognition).
CPU-only via onnxruntime. Models are loaded once at startup.
"""
import os
from typing import List, Dict, Optional, Tuple

import numpy as np
import cv2
from insightface.app import FaceAnalysis


class FaceEngine:
    def __init__(self, model_name: str = "buffalo_l", det_size: int = 640, num_threads: int = 4):
        os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        # ctx_id=-1 -> CPU
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    # ---- image helpers ----
    @staticmethod
    def _decode(data: bytes) -> np.ndarray:
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise ValueError("Could not decode image. Supported: JPEG, PNG, BMP, WEBP.")
        return img

    @staticmethod
    def _face_to_dict(f) -> Dict:
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        kps = getattr(f, "kps", None)
        return {
            "bbox": [x1, y1, x2, y2],
            "det_score": round(float(f.det_score), 4),
            "landmarks": kps.astype(int).tolist() if kps is not None else None,
        }

    @staticmethod
    def _area(f) -> float:
        return (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])

    # ---- public API ----
    def detect(self, data: bytes) -> List[Dict]:
        faces = self.app.get(self._decode(data))
        faces = sorted(faces, key=self._area, reverse=True)
        return [self._face_to_dict(f) for f in faces]

    def embed_largest(self, data: bytes) -> Tuple[Optional[np.ndarray], int]:
        """Return (normed_embedding of the largest face, total faces detected)."""
        faces = self.app.get(self._decode(data))
        if not faces:
            return None, 0
        largest = max(faces, key=self._area)
        return largest.normed_embedding, len(faces)
