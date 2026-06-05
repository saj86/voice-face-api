"""Pure functions for face matching — no insightface/onnx dependency."""
import numpy as np


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two embedding vectors.

    InsightFace `normed_embedding` is already L2-normalized, so this reduces to
    a dot product, but we normalize defensively in case raw embeddings are passed.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def is_match(similarity: float, threshold: float) -> bool:
    return bool(similarity >= threshold)
