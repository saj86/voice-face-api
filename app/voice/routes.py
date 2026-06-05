import os
import json
import shutil
import tempfile
import logging

from fastapi import APIRouter, File, UploadFile, Form, Query, HTTPException

from app.engines import engines
from app.voice.analyze import analyze_call, DEFAULT_ANALYSIS_LANG

log = logging.getLogger("voice")
router = APIRouter()


def _save_temp(file: UploadFile) -> str:
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return tmp.name


@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    num_speakers: int = Query(-1, description="Known speaker count (e.g. 2) — big accuracy win; -1 = auto"),
    language: str | None = Query(None, description="Force ISO code e.g. 'de','fr'; None = auto-detect"),
    threshold: float | None = Query(None, ge=0.0, le=1.0, description="Diarization sensitivity; higher = fewer speakers"),
    initial_prompt: str | None = Query(None, description="Optional context/vocabulary hint to improve accuracy"),
    diarization_mode: str = Query("auto", description="auto | channel | cluster"),
):
    if engines.voice is None:
        raise HTTPException(503, "Voice models still loading; try again shortly.")
    path = _save_temp(file)
    try:
        return engines.voice.transcribe(
            path, num_speakers=num_speakers, language=language or None,
            threshold=threshold, initial_prompt=initial_prompt or None, diarization_mode=diarization_mode,
        )
    except Exception as exc:
        log.exception("transcription failed")
        raise HTTPException(500, str(exc))
    finally:
        os.unlink(path)


@router.post("/analysis")
async def analysis(
    file: UploadFile | None = File(None),
    transcript: str | None = Form(None, description="A transcript JSON (from /transcribe) to skip ASR"),
    num_speakers: int = Query(-1),
    language: str | None = Query(None),
    threshold: float | None = Query(None, ge=0.0, le=1.0),
    initial_prompt: str | None = Query(None),
    diarization_mode: str = Query("auto"),
    output_language: str = Query(DEFAULT_ANALYSIS_LANG, description="Result language; default German. 'auto' matches the call, or pass 'de','fr','en'..."),
):
    """Post-call analysis. Provide EITHER an audio `file` (transcribed first) OR a
    `transcript` JSON string. Returns the transcript plus an `analysis` block."""
    if transcript:
        try:
            transcript_obj = json.loads(transcript)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"Invalid transcript JSON: {e}")
    elif file is not None:
        if engines.voice is None:
            raise HTTPException(503, "Voice models still loading; try again shortly.")
        path = _save_temp(file)
        try:
            transcript_obj = engines.voice.transcribe(
                path, num_speakers=num_speakers, language=language or None,
                threshold=threshold, initial_prompt=initial_prompt or None, diarization_mode=diarization_mode,
            )
        except Exception as exc:
            log.exception("transcription failed")
            raise HTTPException(500, str(exc))
        finally:
            os.unlink(path)
    else:
        raise HTTPException(400, "Provide either an audio 'file' or a 'transcript' JSON.")

    try:
        analysis_block = analyze_call(transcript_obj, engines.analyzer, output_language=output_language)
    except Exception as exc:
        log.exception("analysis failed")
        raise HTTPException(500, str(exc))

    return {
        "transcript": transcript_obj,
        "analysis": analysis_block,
        "llm_enabled": engines.analyzer is not None,
    }
