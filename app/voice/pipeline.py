"""
Speech-to-text + speaker diarization pipeline.

ASR:          faster-whisper  (CTranslate2, CPU, int8)
Diarization:  - dual-channel calls (stereo, uncorrelated channels) -> split by channel
              - everything else -> sherpa-onnx embedding clustering

Both run fully offline once models are present on disk.
"""
import os
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import soxr
import sherpa_onnx
from faster_whisper import WhisperModel

from app.voice.merge import merge_words, group_words
from app.voice.diarize_channel import is_dual_channel, assign_word_channels
from app.voice.voicefeatures import analyze_voice

log = logging.getLogger("pipeline")

DEFAULT_THRESHOLD = float(os.environ.get("DIAR_THRESHOLD", "0.5"))
DEFAULT_MIN_ON = float(os.environ.get("DIAR_MIN_ON", "0.3"))
DEFAULT_MIN_OFF = float(os.environ.get("DIAR_MIN_OFF", "0.5"))
# below this inter-channel correlation, a stereo file is treated as a separated
# dual-channel call (one party per channel)
STEREO_CORR = float(os.environ.get("DIAR_STEREO_CORR", "0.5"))


class Transcriber:
    def __init__(self, whisper_model, segmentation_model, embedding_model,
                 num_threads=4, compute_type="int8"):
        for path in (segmentation_model, embedding_model):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Model not found: {path}")
        self.asr = WhisperModel(whisper_model, device="cpu",
                                compute_type=compute_type, cpu_threads=num_threads)
        self._seg = segmentation_model
        self._emb = embedding_model
        self._threads = num_threads
        self._diar_cache: Dict[Tuple[int, float], "sherpa_onnx.OfflineSpeakerDiarization"] = {}

    # ---------------- diarization (clustering) ----------------
    def _build_diarizer(self, num_speakers, threshold):
        key = (num_speakers, round(threshold, 3))
        if key in self._diar_cache:
            return self._diar_cache[key]
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=self._seg),
                num_threads=self._threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=self._emb, num_threads=self._threads),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_speakers if num_speakers and num_speakers > 0 else -1,
                threshold=threshold,
            ),
            min_duration_on=DEFAULT_MIN_ON, min_duration_off=DEFAULT_MIN_OFF,
        )
        if not config.validate():
            raise RuntimeError("Invalid diarization config — check model paths.")
        diar = sherpa_onnx.OfflineSpeakerDiarization(config)
        self._diar_cache[key] = diar
        return diar

    # ---------------- audio decoding ----------------
    def _decode_full(self, path: str, target_sr: int) -> np.ndarray:
        """Return float32 audio of shape (n, channels) at target_sr.
        Tries soundfile (wav/flac/ogg), falls back to PyAV/ffmpeg (mp3/m4a/...)."""
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)  # (n, ch)
            if sr != target_sr:
                data = soxr.resample(data, sr, target_sr).astype(np.float32)
                if data.ndim == 1:
                    data = data[:, None]
            return data
        except Exception:
            log.info("soundfile could not read %s; decoding with PyAV", path)
            return self._decode_with_av(path, target_sr)

    @staticmethod
    def _decode_with_av(path: str, target_sr: int) -> np.ndarray:
        import av
        container = av.open(path)
        try:
            stream = container.streams.audio[0]
            ch = stream.codec_context.channels or 1
            layout = "stereo" if ch >= 2 else "mono"
            nch = 2 if ch >= 2 else 1
            resampler = av.audio.resampler.AudioResampler(format="flt", layout=layout, rate=target_sr)
            chunks: List[np.ndarray] = []
            for frame in container.decode(stream):
                out = resampler.resample(frame)
                frames = out if isinstance(out, list) else [out]
                for rf in frames:
                    if rf is not None:
                        chunks.append(rf.to_ndarray().reshape(-1))  # interleaved
        finally:
            container.close()
        if not chunks:
            return np.zeros((0, 1), dtype=np.float32)
        flat = np.concatenate(chunks).astype(np.float32)
        return flat.reshape(-1, nch) if nch > 1 else flat[:, None]

    # ---------------- ASR ----------------
    def _asr_words(self, path, language, initial_prompt):
        segments, info = self.asr.transcribe(
            path, word_timestamps=True, vad_filter=True,
            language=language, initial_prompt=initial_prompt,
        )
        words: List[Dict] = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    words.append({"start": w.start, "end": w.end, "word": w.word,
                                  "prob": getattr(w, "probability", None)})
            else:
                words.append({"start": seg.start, "end": seg.end,
                              "word": " " + seg.text.strip(), "prob": None})
        return words, info

    # ---------------- main ----------------
    def transcribe(self, path, num_speakers=-1, language=None,
                   threshold=None, initial_prompt=None, diarization_mode="auto") -> Dict:
        sr = 16000
        words, info = self._asr_words(path, language, initial_prompt)
        audio = self._decode_full(path, sr)              # (n, ch)
        n = audio.shape[0]
        channels = audio.shape[1]

        use_channel = (
            diarization_mode == "channel"
            or (diarization_mode == "auto" and channels >= 2 and is_dual_channel(audio, STEREO_CORR))
        )

        if use_channel and channels >= 2:
            speakers = assign_word_channels(words, audio, sr)
            segments = group_words(words, speakers)
            method = "channel"
            thr = None
        else:
            thr = DEFAULT_THRESHOLD if threshold is None else float(threshold)
            mono = audio.mean(axis=1).astype(np.float32)
            diarizer = self._build_diarizer(num_speakers, thr)
            diar_result = diarizer.process(mono).sort_by_start_time()
            turns = [{"start": r.start, "end": r.end, "speaker": r.speaker} for r in diar_result]
            segments = merge_words(words, turns)
            method = "cluster"

        probs = [w["prob"] for w in words if w.get("prob") is not None]
        confidence = round(sum(probs) / len(probs), 3) if probs else None

        # acoustic voice analysis (pitch / loudness / interruptions / pacing)
        try:
            voice = analyze_voice(audio, sr, segments, method)
        except Exception:
            log.exception("voice feature extraction failed")
            voice = None

        return {
            "language": info.language,
            "duration": round(float(n) / sr, 2),
            "num_speakers": len({s["speaker"] for s in segments}),
            "confidence": confidence,
            "diarization": {
                "method": method,
                "channels": channels,
                "requested_speakers": num_speakers,
                "threshold": thr,
            },
            "voice": voice,
            "segments": segments,
        }
