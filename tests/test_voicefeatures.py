"""Tests for the pure acoustic voice-feature helpers."""
import numpy as np
from app.voice.voicefeatures import (
    rms_dbfs, estimate_pitch, voiced_ratio, overlap_stats, pacing_stats, analyze_voice
)

SR = 16000


def _tone(freq, secs, amp=0.3):
    t = np.arange(int(secs * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _voiced(f0, secs, amp=0.3):
    """Harmonic-rich, speech-like signal (fundamental + harmonics)."""
    t = np.arange(int(secs * SR)) / SR
    sig = sum((1.0 / k) * np.sin(2 * np.pi * k * f0 * t) for k in range(1, 6))
    return (amp * sig / np.max(np.abs(sig))).astype(np.float32)


def test_dbfs_fullscale():
    assert abs(rms_dbfs(_tone(200, 1.0, amp=1.0)) - (-3.0)) < 1.0  # sine RMS ~ -3 dBFS


def test_pitch_estimate():
    med, p10, p90 = estimate_pitch(_voiced(150, 1.5), SR)
    assert abs(med - 150) < 12


def test_voiced_ratio_silence_vs_tone():
    assert voiced_ratio(np.zeros(SR, dtype=np.float32), SR) == 0.0
    assert voiced_ratio(_voiced(150, 1.0), SR) > 0.8


def test_overlap_stats():
    n = 6 * SR
    L = np.zeros(n, dtype=np.float32); R = np.zeros(n, dtype=np.float32)
    L[: 4 * SR] = _tone(120, 4)            # left talks 0-4s
    R[3 * SR : 5 * SR] = _tone(220, 2)     # right talks 3-5s  -> 1s overlap (3-4s)
    events, secs = overlap_stats(np.stack([L, R], axis=1), SR)
    assert events == 1 and 0.6 < secs < 1.6


def test_pacing_stats():
    segs = [
        {"start": 0.0, "end": 3.0, "speaker": "Speaker 1"},
        {"start": 4.0, "end": 5.0, "speaker": "Speaker 2"},  # 1s gap after S1
        {"start": 5.0, "end": 6.5, "speaker": "Speaker 1"},
    ]
    p = pacing_stats(segs)
    assert p["longest_monologue"]["Speaker 1"] == 3.0
    assert p["avg_response_gap"] >= 0.0


def test_analyze_voice_channel_mode():
    n = 6 * SR
    L = np.zeros(n, dtype=np.float32); R = np.zeros(n, dtype=np.float32)
    L[: 3 * SR] = _voiced(120, 3)     # speaker 1 low pitch on left
    R[3 * SR :] = _voiced(230, 3)     # speaker 2 higher pitch on right
    stereo = np.stack([L, R], axis=1)
    segs = [{"start": 0.5, "end": 2.5, "speaker": "Speaker 1"},
            {"start": 3.5, "end": 5.5, "speaker": "Speaker 2"}]
    out = analyze_voice(stereo, SR, segs, "channel")
    assert out["speakers"]["Speaker 1"]["pitch_hz"] < out["speakers"]["Speaker 2"]["pitch_hz"]


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
