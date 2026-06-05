"""Sanity tests for the pure face-matching logic (no models needed)."""
import numpy as np
from app.face.similarity import cosine_similarity, is_match


def test_identical():
    v = [0.1, 0.2, 0.3, 0.4]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_orthogonal():
    assert abs(cosine_similarity([1, 0], [0, 1])) < 1e-9


def test_opposite():
    assert abs(cosine_similarity([1, 0], [-1, 0]) + 1.0) < 1e-9


def test_zero_vector_safe():
    assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


def test_normed_equivalence():
    rng = np.random.default_rng(0)
    a = rng.normal(size=512); a /= np.linalg.norm(a)
    b = rng.normal(size=512); b /= np.linalg.norm(b)
    assert abs(cosine_similarity(a, b) - float(np.dot(a, b))) < 1e-9


def test_match_threshold():
    assert is_match(0.45, 0.40) is True
    assert is_match(0.39, 0.40) is False
    assert is_match(0.40, 0.40) is True


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except AssertionError as e:
                fails += 1; print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
