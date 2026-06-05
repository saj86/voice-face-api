# Voice + Face API (offline, CPU, single port)

One self-hosted service that does **speech-to-text with speaker diarization**
and **face detection + 1:1 face comparison**. It serves a test **UI at `/`** and
exposes both APIs under sub-paths of the **same address and port** — no second
port, no CORS setup, no separate UI to host. Runs **fully offline on CPU**; all
models are baked into the Docker image at build time.

```
http://localhost:8000/                  ← test UI (open in a browser)
http://localhost:8000/examples          ← copy-paste code samples (C#, JS, cURL)http://localhost:8000/docs              ← interactive OpenAPI docs
POST http://localhost:8000/voice/transcribe
POST http://localhost:8000/voice/analysis
POST http://localhost:8000/face/detect
POST http://localhost:8000/face/compare
GET  http://localhost:8000/health
```

| Capability    | Library          | Model (baked in)                         |
|---------------|------------------|------------------------------------------|
| Transcription | `faster-whisper` | Whisper `small` (multilingual)           |
| Diarization   | `sherpa-onnx`    | pyannote segmentation + TitaNet embedding|
| Face detect   | `insightface`    | SCRFD (`buffalo_l` pack)                 |
| Face compare  | `insightface`    | ArcFace (`buffalo_l` pack)               |

## Run

```bash
docker compose up --build      # first build downloads models into the image
```
Then open `http://localhost:8000`. After startup (models take a moment to load),
`GET /health` reports `voice_ready` and `face_ready`.

## Why one port?

The previous draft ran two containers on ports 8000/8001. This version runs a
single FastAPI app that mounts the voice routes under `/voice`, the face routes
under `/face`, and returns the UI at `/`. Because the UI is now served from the
same origin as the APIs, it calls relative paths (`/voice/transcribe`, …) and
needs no endpoint configuration. If you ever want to scale the two workloads
independently, splitting them back into separate services is the time to add a
reverse proxy (nginx/Traefik) to keep them under one address.

## Test console (the UI at `/`)

The Call Analysis tab supports **multiple files at once** (and a **`.zip`** —
audio entries are extracted in-browser via the bundled JSZip), shows a **queue
with per-file status** (queued → transcribing → analyzing → done/error), and
exposes the controls above: number of speakers, spoken language, results
language, and diarization sensitivity. Click a finished file to open a
**two-pane result**: a sticky **left panel** with the waveform player (click to
seek) and live spectrum, and a **right panel with Transcript / Analysis tabs**, so
you can read the transcript while the player stays in view. Segments are clickable
and highlight as the audio plays. The waveform and spectrum use
the browser's Web Audio API (no external libraries), so the console stays
offline. (Fonts load from Google Fonts as progressive enhancement; without
internet the UI still works with system fonts.)


### Console UI
- **Multilingual interface** (EN/DE/FR/ES/IT) via the language selector top-right; analysis **tags are localized** to the chosen language while machine values stay canonical.
- **Company logo:** drop a file at `app/ui/logo.png` (served at `/static/logo.png`); a monogram is shown if absent. Brand name is the `#brand-name` element in `app/ui/index.html`.
- **Scrollable** fixed-height Transcript & Analysis panels; sticky player; **sentiment timeline** strip; **per-line emotion** tags; **keyword alerts** (click to seek); **export** JSON / CSV / Print-to-PDF.

## API reference

### `POST /voice/transcribe`
`multipart/form-data`, field **`file`** = audio (WAV/MP3/M4A/FLAC/OGG — non-WAV
is decoded via the bundled ffmpeg). Query params:
- `num_speakers` — **set this when you know it** (e.g. `2` for a normal call).
  Fixing the count is the single biggest diarization-accuracy win; `-1` = auto.
- `language` — force an ISO code (`de`, `fr`, …). Forcing the language helps a
  lot on short or noisy audio where auto-detect drifts.
- `threshold` — diarization sensitivity, 0–1. **Higher = fewer speakers.** Raise
  it if the call is being split into too many speakers; lower it if distinct
  speakers are being merged. Default `0.5`.
- `initial_prompt` — optional context/vocabulary hint (names, product terms) to
  bias transcription.
```json
{ "language": "de", "duration": 213.9, "num_speakers": 2, "confidence": 0.79,
  "diarization": {"requested_speakers": 2, "threshold": 0.5},
  "segments": [ {"start":2.7,"end":10.4,"speaker":"Speaker 1","text":"..."} ] }
```

### `POST /voice/analysis`
Post-call analysis (hybrid: deterministic metrics + a local LLM). Send **either**
an audio `file` (it gets transcribed first) **or** a `transcript` JSON string
from `/transcribe` (skips re-transcribing — what the UI does). Query param
`output_language` (`auto` matches the call language, or pass `de`/`fr`/`en`/…),
plus the same `num_speakers`/`language`/`threshold`/`initial_prompt` controls.

`analysis.insights` (LLM) includes: `call_reason`, `call_category`, `urgency`,
`overall_sentiment`, `customer_sentiment`, `sentiment_start`/`sentiment_end`
(progression), `speaker_roles` (agent/customer), `is_complaint` +
`complaint_reason`, `tone`, `agent_professionalism`, `resolution`,
`customer_satisfaction`, `escalation_needed`, `topics`, `key_points`,
`unanswered_questions`, `commitments`, `entities` (products / order-or-article
numbers / people), `summary`, `action_items`, `follow_up_required`. Enum values
stay English; free-text is written in `output_language`, which **defaults to
German** (`ANALYSIS_LANG`, default `de`; `auto` matches the call language).

`analysis.voice` is **acoustic analysis computed from the audio signal** (pure
numpy, offline, no model):
- per speaker: `pitch_hz` (median F0) + `pitch_range`, `loudness_dbfs`,
  `voiced_ratio`;
- interaction: `interruptions` + `overtalk_sec` (true overlap, dual-channel
  only), `avg_response_gap` (responsiveness), `longest_monologue` per speaker.

> Pitch caveat: telephone audio is band-limited (~300–3400 Hz), so a low male
> fundamental is partly filtered out. We use a cepstrum estimator (robust to a
> missing fundamental) but pitch on 8 kHz calls is still approximate. Loudness,
> voiced ratio, interruptions and pacing are reliable.

### Improving accuracy
- **Dual-channel calls (automatic):** many contact-center recordings are stereo
  with each party on its own channel. The pipeline detects this (two channels,
  low inter-channel correlation) and **splits speakers by channel** — near-perfect
  and far better than clustering on downmixed 8 kHz mono. The response shows
  `diarization.method` = `channel` or `cluster`. Force it with `diarization_mode`
  (`auto` | `channel` | `cluster`); tune the correlation cutoff via
  `DIAR_STEREO_CORR` (default 0.5).
- **Mono / single-channel:** set `num_speakers` (UI "Speakers" field) — for
  two-party calls, `2` fixes most over-/under-segmentation. Otherwise tune
  `threshold` (env `DIAR_THRESHOLD`, default 0.5): **higher merges more speakers
  (fewer total), lower splits more**. The default embedding model is
  English-trained; for hard multilingual mono audio, swapping `embedding.onnx`
  for a larger/multilingual sherpa-onnx model helps most.
- **Transcription on low-quality audio:** force `language`, build a larger
  `WHISPER_MODEL` (`medium` is markedly better on noisy/non-English speech), and
  pass an `initial_prompt` with expected names/terms.

```json
{
  "transcript": { ...same shape as /transcribe... },
  "analysis": {
    "confidence": 0.79, "language": "de",
    "metrics": { "duration": 213.9, "total_speech": 165.7, "total_silence": 48.2,
      "longest_silence": 31.5, "num_turns": 17, "num_speakers": 2,
      "speakers": { "Speaker 1": {"talk_ratio": 0.13, "words_per_min": 86.4, "...": "..."} } },
    "insights": { "overall_sentiment": "neutral", "is_complaint": false,
      "resolution": "follow-up required", "customer_satisfaction": "medium",
      "summary": "...", "action_items": ["..."], "entities": {"order_or_article_numbers": ["444-877"]} }
  },
  "llm_enabled": true
}
```
`metrics` and `confidence` are always computed (cheap, deterministic). `insights`
comes from the local LLM and is `null` if the LLM is disabled (`ENABLE_LLM=0`).

#### The analysis LLM
A small local instruct model runs the language understanding (sentiment,
complaint, tone, topics, summary, action items) in one pass via `llama.cpp`,
fully offline on CPU. Default: **Qwen2.5-1.5B-Instruct (Apache-2.0)**, ~1 GB.

- **Licensing:** Qwen2.5 `0.5B / 1.5B / 7B / 14B / 32B` are Apache-2.0;
  **3B and 72B are NOT** (qwen-research, non-commercial) — avoid them for a
  commercial product. Override the model with build args:
  `--build-arg LLM_REPO=Qwen/Qwen2.5-7B-Instruct-GGUF --build-arg LLM_FILE=qwen2.5-7b-instruct-q4_k_m.gguf`
  for higher accuracy (slower, ~4.7 GB).
- **Disable it:** set `ENABLE_LLM=0` to return metrics + confidence only.
- **Summary language:** written in English by default regardless of call
  language (change the prompt in `app/voice/analyze.py`).
- It's the heaviest component — first analysis call is slowest as the model warms.
`multipart/form-data`, field **`file`** = an image. Returns faces largest-first
with `bbox` `[x1,y1,x2,y2]`, `det_score`, and `landmarks`.

### `POST /face/compare`
`multipart/form-data`, fields **`file1`**, **`file2`**. Optional query
`threshold` (0–1, default 0.40).
```json
{ "similarity": 0.61, "match": true, "threshold": 0.40,
  "faces_detected": {"image1":1,"image2":1} }
```

## European / non-English languages

Transcription is multilingual out of the box — Whisper `small` (the default
here) auto-detects the language, or you can force it with `?language=de` /
`fr` / `es` / `it` / …. For higher accuracy build with `WHISPER_MODEL=medium`
(slower). Diarization is language-independent (voice characteristics, not
words), so it needs no change.

## Adding an API token for gated/private model downloads

The bundled models need **no** token. If you swap in a *gated* model (e.g.
pyannote on Hugging Face) or pull from a private registry, the download happens
at **build time** and needs a token. Don't use a plain `ARG`/`ENV` for secrets —
they get baked into image layers and leak via `docker history`. Use a **BuildKit
build secret**, which is mounted only for one `RUN` step and never persists:

```bash
echo "hf_your_token_here" > hf_token.txt
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,src=hf_token.txt \
  -t voice-face-api .
```

The Dockerfile already starts with `# syntax=docker/dockerfile:1` and contains a
commented, ready-to-uncomment step:

```dockerfile
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token)" \
    HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    python -c "from huggingface_hub import snapshot_download; \
               snapshot_download('pyannote/segmentation-3.0', local_dir='/models/pyannote')"
```

You can also pass the secret straight from an env var:
`docker build --secret id=hf_token,env=HF_TOKEN .`.

### Using a `.env` file

Storing the token in `.env` is the right move — but `.env` is *storage*, not a
delivery mechanism, and **where** you use it matters:

- **Build-time (offline image, this project's default):** keep the value in
  `.env`, but it must reach the build through a **BuildKit secret**, never a
  build `ARG`/`ENV` — args are baked into image layers and leak via
  `docker history`. `docker-compose.yml` is already wired for this: the
  top-level `secrets.hf_token` reads `HF_TOKEN` from your environment/`.env`,
  and `build.secrets` exposes it to the (commented) gated-download step. So:
  ```bash
  cp .env.example .env          # then set HF_TOKEN=...
  DOCKER_BUILDKIT=1 docker compose build
  ```
- **Runtime (model fetched on first use, not offline):** passing the token to
  the *container* via `.env` → `environment:`/`env_file:` is fine and safe
  enough — it lives in container env, not image layers. In that case also remove
  the `HF_HUB_OFFLINE=1` line so the download is allowed.

Add `.env` to `.gitignore` (already done here) so the token is never committed.

> Note: with the commercial-friendly recognition models below, you likely won't
> need a gated model or token at all.

## Commercial-use face recognition models

The default `buffalo_l` recognition weights are **non-commercial / research
only** (the InsightFace *code* is MIT, but its model packs are not). For
commercial use, swap to permissively-licensed weights:

| Option            | License    | Notes                                                        |
|-------------------|------------|--------------------------------------------------------------|
| YuNet + SFace     | Apache 2.0 | ~40 MB, built into OpenCV, CPU-friendly, no token. Lower accuracy than ArcFace but commercial-safe. |
| AuraFace-v1       | Apache 2.0 | ResNet100 ArcFace rebuild, ~99.6% LFW, near drop-in for the current InsightFace code. Larger/slower on CPU. |
| dlib `face_recognition` | permissive | Easy, CPU-friendly, but lower accuracy and a heavier build. |

`buffalo_l` is kept as the default only for out-of-the-box accuracy in testing.
Switch before any commercial deployment.

## Face authentication — important caveats

- **No liveness / anti-spoofing.** A printed photo or a face on a screen can
  pass `/compare`. Add a liveness check before using this for real auth.
- **Tune the threshold** (default 0.40; typical 1:1 range 0.30–0.45) on your own
  genuine/impostor pairs.
- **`buffalo_l` license** is research / non-commercial. Review before shipping a
  commercial product; this is a licensing fact to check, not legal advice.
- Face embeddings are biometric personal data (GDPR etc. apply in Europe).

## Resource note

Both model stacks plus the analysis LLM load into one process. With `small`
Whisper + `buffalo_l` + Qwen2.5-1.5B that's roughly ~4 GB RAM; a 7B LLM pushes
it higher. Set `NUM_THREADS` to your core count. To shrink: use a smaller
Whisper (`base`), the `buffalo_s` face pack, or `ENABLE_LLM=0`. The build also
gets larger and slower because it now bakes in the GGUF model.

## Layout

```
voice-face-api/
├── app/
│   ├── main.py            # single entrypoint: UI + routers + startup
│   ├── engines.py         # shared loaded-model holder
│   ├── voice/{pipeline,merge,routes}.py
│   ├── voice/analyze.py   # metrics + local-LLM call analysis
│   ├── face/{faceapi,similarity,routes}.py
│   └── ui/index.html      # combined test console
│       └── vendor/jszip.min.js  # bundled (offline ZIP extraction)
├── tests/                 # pure-logic unit tests
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Local dev (no Docker)

```bash
pip install -r requirements.txt        # + system libs: libsndfile1 libgl1 libglib2.0-0
# fetch the sherpa models into ./models (see the wget lines in the Dockerfile)
MODELS_DIR=./models uvicorn app.main:app --reload --port 8000
PYTHONPATH=. python tests/test_merge.py && PYTHONPATH=. python tests/test_similarity.py
```
# voice-face-api
