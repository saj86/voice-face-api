"""
Channel-based diarization for dual-channel call recordings.

Many contact-center recordings are stereo with each party on its own channel
(agent left, customer right). When that's the case the channels are essentially
uncorrelated, and we can assign speakers by *which channel is active* during each
word — far more accurate than clustering embeddings on a downmixed 8 kHz mono
signal. These helpers are pure (numpy only) so they can be unit-tested.
"""
from typing import List, Dict

import numpy as np


def channel_correlation(stereo: np.ndarray) -> float:
    """Pearson correlation between the two channels of an (n, 2) array."""
    if stereo.ndim < 2 or stereo.shape[1] < 2:
        return 1.0
    left, right = stereo[:, 0], stereo[:, 1]
    if left.std() < 1e-7 or right.std() < 1e-7:
        return 1.0
    return float(np.corrcoef(left, right)[0, 1])


def is_dual_channel(stereo: np.ndarray, corr_threshold: float = 0.5,
                    min_rms: float = 1e-4) -> bool:
    """True if the audio looks like a separated dual-channel call:
    two channels, both carrying audio, and low inter-channel correlation."""
    if stereo.ndim < 2 or stereo.shape[1] < 2:
        return False
    left, right = stereo[:, 0], stereo[:, 1]
    rms_l = float(np.sqrt(np.mean(left ** 2)))
    rms_r = float(np.sqrt(np.mean(right ** 2)))
    if rms_l < min_rms or rms_r < min_rms:
        return False  # one side is silent -> effectively mono
    return abs(channel_correlation(stereo)) < corr_threshold


def assign_word_channels(words: List[Dict], stereo: np.ndarray, sr: int) -> List[int]:
    """For each word, return the channel (0=left, 1=right) with more energy
    during that word's time span. 0 -> Speaker 1, 1 -> Speaker 2."""
    if not words:
        return []
    left_sq = stereo[:, 0] ** 2
    right_sq = stereo[:, 1] ** 2
    n = len(left_sq)
    out: List[int] = []
    for w in words:
        i0 = max(0, int(w["start"] * sr))
        i1 = min(n, int(w["end"] * sr))
        if i1 <= i0:
            i1 = min(n, i0 + 1)
        el = float(left_sq[i0:i1].mean()) if i1 > i0 else 0.0
        er = float(right_sq[i0:i1].mean()) if i1 > i0 else 0.0
        out.append(0 if el >= er else 1)
    return out
