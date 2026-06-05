"""Tests for dual-channel detection and per-word channel assignment."""
import numpy as np
from app.voice.diarize_channel import is_dual_channel, channel_correlation, assign_word_channels
from app.voice.merge import group_words

SR = 16000


def _make_dual_channel():
    """3s left speech, then 3s right speech — a turn-taking dual-channel call."""
    n = 6 * SR
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(0)
    left[: 3 * SR] = rng.normal(0, 0.2, 3 * SR)   # speaker A on left, 0-3s
    right[3 * SR :] = rng.normal(0, 0.2, 3 * SR)   # speaker B on right, 3-6s
    return np.stack([left, right], axis=1)


def test_detects_dual_channel():
    assert is_dual_channel(_make_dual_channel(), 0.5) is True


def test_mono_is_not_dual():
    n = 4 * SR
    sig = np.random.default_rng(1).normal(0, 0.2, n).astype(np.float32)
    dual_mono = np.stack([sig, sig], axis=1)  # identical channels
    assert is_dual_channel(dual_mono, 0.5) is False
    assert abs(channel_correlation(dual_mono) - 1.0) < 1e-6


def test_one_silent_channel_is_not_dual():
    n = 4 * SR
    sig = np.random.default_rng(2).normal(0, 0.2, n).astype(np.float32)
    silent = np.zeros(n, dtype=np.float32)
    assert is_dual_channel(np.stack([sig, silent], axis=1), 0.5) is False


def test_assign_word_channels_and_group():
    stereo = _make_dual_channel()
    words = [
        {"start": 0.5, "end": 1.5, "word": " hello"},
        {"start": 1.5, "end": 2.5, "word": " there"},
        {"start": 3.5, "end": 4.5, "word": " hi"},
        {"start": 4.5, "end": 5.5, "word": " back"},
    ]
    spk = assign_word_channels(words, stereo, SR)
    assert spk == [0, 0, 1, 1]
    segs = group_words(words, spk)
    assert len(segs) == 2
    assert segs[0]["speaker"] == "Speaker 1" and segs[0]["text"] == "hello there"
    assert segs[1]["speaker"] == "Speaker 2" and segs[1]["text"] == "hi back"


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
