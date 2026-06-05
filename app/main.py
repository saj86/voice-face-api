"""
Unified Voice + Face service — one app, one port.

  GET  /                    -> combined test UI
  GET  /health
  POST /voice/transcribe    -> diarized transcript
  POST /face/detect         -> face bounding boxes
  POST /face/compare        -> 1:1 face similarity / match
  GET  /docs                -> interactive API docs (Swagger)

Both model stacks load once at startup and are shared across requests.
"""
import os
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.engines import engines
from app.voice.pipeline import Transcriber
from app.face.faceapi import FaceEngine
from app.voice.routes import router as voice_router
from app.face.routes import router as face_router

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
UI_FILE = Path(__file__).parent / "ui" / "index.html"

app = FastAPI(title="Voice + Face API", version="1.0.0")

# Same-origin UI doesn't need CORS, but keep it open for external API clients.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_models() -> None:
    threads = int(os.environ.get("NUM_THREADS", "4"))

    log.info("Loading voice models...")
    engines.voice = Transcriber(
        whisper_model=os.environ.get("WHISPER_MODEL", "small"),
        segmentation_model=os.path.join(MODELS_DIR, "segmentation.onnx"),
        embedding_model=os.path.join(MODELS_DIR, "embedding.onnx"),
        num_threads=threads,
        compute_type=os.environ.get("COMPUTE_TYPE", "int8"),
    )

    log.info("Loading face models...")
    engines.face = FaceEngine(
        model_name=os.environ.get("FACE_MODEL", "buffalo_l"),
        det_size=int(os.environ.get("DET_SIZE", "640")),
        num_threads=threads,
    )

    if os.environ.get("ENABLE_LLM", "1") == "1":
        log.info("Loading analysis LLM...")
        try:
            from app.voice.analyze import ConversationAnalyzer

            engines.analyzer = ConversationAnalyzer(
                model_path=os.environ.get("LLM_MODEL_PATH", "/models/llm.gguf"),
                n_threads=threads,
                n_ctx=int(os.environ.get("LLM_CTX", "8192")),
            )
        except Exception:
            log.exception("Could not load analysis LLM; /voice/analysis will return metrics only")
            engines.analyzer = None

    log.info("All models loaded. Ready.")


app.include_router(voice_router, prefix="/voice", tags=["voice"])
app.include_router(face_router, prefix="/face", tags=["face"])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "voice_ready": engines.voice is not None,
        "face_ready": engines.face is not None,
        "llm_ready": engines.analyzer is not None,
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(UI_FILE)


@app.get("/examples", include_in_schema=False)
def examples():
    return FileResponse(Path(__file__).parent / "ui" / "examples.html")


# Serve vendored static assets (e.g. JSZip) so the UI works fully offline.
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "ui")), name="static")
