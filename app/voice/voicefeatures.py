"""
Acoustic voice analysis — offline, numpy only (no ML model, no extra deps).

Derives per-speaker and interaction-level signals from the raw audio + the
diarized segments:
  - pitch (median F0 and range)  -> stress / animation / rough gender cue
  - loudness (dBFS)              -> how loud each party is
  - voiced ratio                 -> how much of their time is actual speech
  - interruptions / overtalk     -> both channels active at once (dual-channel)
  - response gaps & monologues   -> conversational pacing

All functions are pure so they can be unit-tested.
"""
from typing import List, Dict, Tuple

import numpy as np


def rms_dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    if rms <= 1e-9:
        return -120.0
    return 20.0 * np.log10(rms)


def estimate_pitch(x: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 320.0,
                   frame: float = 0.04, hop: float = 0.02) -> Tuple[float, float, float]:
    """Median F0 (Hz) and 10th/90th percentiles over voiced frames, via a
    cepstrum peak. Cepstrum recovers the fundamental even when it's weak/missing
    (as on band-limited telephone audio), unlike plain autocorrelation."""
    n = x.size
    fl = int(frame * sr)
    hl = max(1, int(hop * sr))
    if n < fl or fl < 2:
        return (0.0, 0.0, 0.0)
    min_lag = max(2, int(sr / fmax))
    max_lag = int(sr / fmin)
    nfft = 1 << int(np.ceil(np.log2(fl)))
    window = np.hanning(fl)
    peak_amp = float(np.max(np.abs(x))) + 1e-9
    pitches: List[float] = []
    for s in range(0, n - fl, hl):
        f = x[s:s + fl].astype(np.float64)
        if np.sqrt(np.mean(f ** 2)) < 0.06 * peak_amp:  # skip quiet/unvoiced
            continue
        spec = np.fft.rfft(f * window, nfft)
        logmag = np.log(np.abs(spec) + 1e-8)
        cep = np.fft.irfft(logmag)
        hi = min(max_lag, len(cep) - 1)
        if hi <= min_lag:
            continue
        seg = cep[min_lag:hi]
        if seg.size == 0:
            continue
        lag = min_lag + int(np.argmax(seg))
        # voicing: cepstral peak must stand out from the local mean
        if cep[lag] < 3.0 * (np.mean(np.abs(seg)) + 1e-9):
            continue
        pitches.append(sr / lag)
    if not pitches:
        return (0.0, 0.0, 0.0)
    p = np.array(pitches)
    # octave-correction: if many frames are ~2x the low cluster, fold them down
    lo = np.median(p[p <= np.percentile(p, 40)]) if p.size else 0.0
    if lo > 0:
        p = np.where(p > 1.6 * lo, p / 2.0, p)
    return (float(np.median(p)), float(np.percentile(p, 10)), float(np.percentile(p, 90)))


def voiced_ratio(x: np.ndarray, sr: int, frame: float = 0.03) -> float:
    n = x.size
    fl = int(frame * sr)
    if n < fl or fl < 1:
        return 0.0
    peak = float(np.max(np.abs(x))) + 1e-9
    voiced = total = 0
    for s in range(0, n - fl, fl):
        total += 1
        if np.sqrt(np.mean(x[s:s + fl] ** 2)) > 0.05 * peak:
            voiced += 1
    return voiced / total if total else 0.0


def overlap_stats(stereo: np.ndarray, sr: int, frame: float = 0.05,
                  min_event: float = 0.3) -> Tuple[int, float]:
    """Count interruption events and total seconds where BOTH channels are
    active simultaneously. Only meaningful for separated dual-channel audio."""
    if stereo.ndim < 2 or stereo.shape[1] < 2:
        return (0, 0.0)
    fl = max(1, int(frame * sr))
    L, R = stereo[:, 0], stereo[:, 1]
    peakL = float(np.max(np.abs(L))) + 1e-9
    peakR = float(np.max(np.abs(R))) + 1e-9
    both = []
    for s in range(0, stereo.shape[0] - fl, fl):
        al = np.sqrt(np.mean(L[s:s + fl] ** 2)) > 0.08 * peakL
        ar = np.sqrt(np.mean(R[s:s + fl] ** 2)) > 0.08 * peakR
        both.append(al and ar)
    # collapse contiguous "both active" frames into events >= min_event seconds
    events = 0
    total = 0.0
    run = 0
    for b in both + [False]:
        if b:
            run += 1
        else:
            dur = run * frame
            if dur >= min_event:
                events += 1
                total += dur
            run = 0
    return (events, total)


def pacing_stats(segments: List[Dict]) -> Dict:
    ordered = sorted(segments, key=lambda s: s["start"])
    gaps = [b["start"] - a["end"] for a, b in zip(ordered, ordered[1:])
            if b["speaker"] != a["speaker"] and b["start"] > a["end"]]
    longest: Dict[str, float] = {}
    for s in ordered:
        d = s["end"] - s["start"]
        longest[s["speaker"]] = max(longest.get(s["speaker"], 0.0), d)
    return {
        "avg_response_gap": round(float(np.mean(gaps)), 2) if gaps else 0.0,
        "longest_monologue": {k: round(v, 1) for k, v in longest.items()},
    }


def analyze_voice(audio2d: np.ndarray, sr: int, segments: List[Dict], method: str) -> Dict:
    """Assemble per-speaker acoustic features + interaction metrics."""
    if audio2d.ndim == 1:
        audio2d = audio2d[:, None]
    mono = audio2d.mean(axis=1).astype(np.float32)
    labels = sorted({s["speaker"] for s in segments})

    speakers: Dict[str, Dict] = {}
    for lbl in labels:
        # pick the source signal: own channel (dual-channel) or the mono mix
        if method == "channel" and audio2d.shape[1] >= 2:
            try:
                idx = int(lbl.split()[-1]) - 1
            except Exception:
                idx = 0
            src = audio2d[:, idx] if 0 <= idx < audio2d.shape[1] else mono
        else:
            src = mono
        # only the times this speaker is actually talking (avoids silence/crosstalk)
        parts = [src[int(s["start"] * sr):int(s["end"] * sr)]
                 for s in segments if s["speaker"] == lbl]
        sig = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        med, p10, p90 = estimate_pitch(sig, sr)
        speakers[lbl] = {
            "pitch_hz": round(med, 1),
            "pitch_range": [round(p10, 1), round(p90, 1)],
            "loudness_dbfs": round(rms_dbfs(sig), 1),
            "voiced_ratio": round(voiced_ratio(sig, sr), 2),
        }

    interaction: Dict = {}
    if method == "channel" and audio2d.shape[1] >= 2:
        events, secs = overlap_stats(audio2d, sr)
        interaction["interruptions"] = events
        interaction["overtalk_sec"] = round(secs, 2)
    interaction.update(pacing_stats(segments))

    return {"speakers": speakers, "interaction": interaction}
