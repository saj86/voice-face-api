"""
Pure functions that combine ASR word timings with diarization speaker turns.

Kept free of heavy imports (no whisper / sherpa) so the core logic can be
unit-tested on its own.
"""
from typing import List, Dict, Optional


def assign_speaker(word: Dict, turns: List[Dict]) -> int:
    """Return the speaker id of the diarization turn that overlaps `word` most.

    Falls back to the nearest turn (by midpoint) if there is no overlap, and to
    speaker 0 if there are no turns at all.
    """
    if not turns:
        return 0

    ws, we = word["start"], word["end"]
    best_speaker: Optional[int] = None
    best_overlap = 0.0

    for t in turns:
        overlap = min(we, t["end"]) - max(ws, t["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = t["speaker"]

    if best_speaker is not None:
        return best_speaker

    # No overlap: snap to the closest turn by midpoint distance.
    mid = (ws + we) / 2.0
    return min(turns, key=lambda t: abs((t["start"] + t["end"]) / 2.0 - mid))["speaker"]


def merge_words(words: List[Dict], turns: List[Dict]) -> List[Dict]:
    """Group consecutive words sharing a speaker into transcript segments.

    Each input word is {"start", "end", "word"}.
    Each turn is {"start", "end", "speaker"} (speaker is an int id).
    """
    if not words:
        return []
    speakers = [assign_speaker(w, turns) for w in words]
    return group_words(words, speakers)


def group_words(words: List[Dict], speakers: List[int]) -> List[Dict]:
    """Group consecutive words that share a speaker id into segments.

    `speakers[i]` is the speaker id for `words[i]`. Output segments are
    {"start", "end", "speaker", "text"} with a 1-indexed "Speaker N" label.
    """
    if not words:
        return []

    segments: List[Dict] = []
    current = None

    for w, spk in zip(words, speakers):
        if current is not None and current["_spk"] == spk:
            current["end"] = w["end"]
            current["text"] += w["word"]
        else:
            if current is not None:
                segments.append(_finalize(current))
            current = {"_spk": spk, "start": w["start"], "end": w["end"], "text": w["word"]}

    if current is not None:
        segments.append(_finalize(current))

    return segments


def _finalize(seg: Dict) -> Dict:
    return {
        "start": round(seg["start"], 2),
        "end": round(seg["end"], 2),
        "speaker": f"Speaker {seg['_spk'] + 1}",
        "text": seg["text"].strip(),
    }
