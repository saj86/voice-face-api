# syntax=docker/dockerfile:1
# Unified Voice + Face API — offline, CPU-only, single port (8000).
# All models (ASR, diarization, face, and the analysis LLM) are downloaded at
# BUILD time and baked into the image.
FROM python:3.11-slim

ARG WHISPER_MODEL=small
# Analysis LLM (GGUF). Default is Apache-2.0 and CPU-friendly.
# NOTE: Qwen2.5-3B and 72B are NOT Apache (qwen-research license) — avoid for commercial use.
# Apache-2.0 sizes: 0.5B, 1.5B, 7B, 14B, 32B.  7B = better quality, slower on CPU.
ARG LLM_REPO=Qwen/Qwen2.5-1.5B-Instruct-GGUF
ARG LLM_FILE=qwen2.5-1.5b-instruct-q4_k_m.gguf

ENV WHISPER_MODEL=${WHISPER_MODEL} \
    LLM_REPO=${LLM_REPO} \
    LLM_FILE=${LLM_FILE} \
    LLM_MODEL_PATH=/models/llm.gguf \
    ENABLE_LLM=1 \
    ANALYSIS_LANG=de \
    FACE_MODEL=buffalo_l \
    MATCH_THRESHOLD=0.40 \
    DET_SIZE=640 \
    NUM_THREADS=4 \
    COMPUTE_TYPE=int8 \
    MODELS_DIR=/models \
    HF_HOME=/models/hf \
    INSIGHTFACE_HOME=/root/.insightface \
    PYTHONUNBUFFERED=1

# System libs:
#   libsndfile1            -> soundfile (voice)
#   libgl1, libglib2.0-0   -> OpenCV (face)
#   build-essential, cmake -> build insightface ext + llama-cpp-python
#   wget, bzip2            -> fetching diarization models
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 libgl1 libglib2.0-0 build-essential cmake \
        wget bzip2 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Diarization models (sherpa-onnx, non-gated) ---
RUN mkdir -p ${MODELS_DIR} && cd /tmp \
 && wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 \
 && tar xjf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 \
 && cp sherpa-onnx-pyannote-segmentation-3-0/model.onnx ${MODELS_DIR}/segmentation.onnx \
 && wget -q -O ${MODELS_DIR}/embedding.onnx \
      https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx \
 && rm -rf /tmp/*

# --- Whisper ASR model ---
RUN python -c "import os; from faster_whisper import WhisperModel; \
WhisperModel(os.environ['WHISPER_MODEL'], device='cpu', compute_type='int8')"

# --- Face model pack (InsightFace) ---
RUN python -c "from insightface.app import FaceAnalysis; \
a=FaceAnalysis(name='buffalo_l', allowed_modules=['detection','recognition'], providers=['CPUExecutionProvider']); \
a.prepare(ctx_id=-1, det_size=(640,640))"

# --- Analysis LLM (GGUF) -> /models/llm.gguf ---
RUN python -c "import os,shutil; from huggingface_hub import hf_hub_download; \
p=hf_hub_download(repo_id=os.environ['LLM_REPO'], filename=os.environ['LLM_FILE']); \
shutil.copy(p, os.environ['LLM_MODEL_PATH'])"

# From here on, force fully-offline behaviour at runtime.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY app ./app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
