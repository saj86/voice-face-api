import os
import logging

from fastapi import APIRouter, File, UploadFile, Query, HTTPException

from app.engines import engines
from app.face.similarity import cosine_similarity, is_match

log = logging.getLogger("face")
router = APIRouter()

DEFAULT_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.40"))


@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    if engines.face is None:
        raise HTTPException(503, "Face models still loading; try again shortly.")
    try:
        faces = engines.face.detect(await file.read())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"count": len(faces), "faces": faces}


@router.post("/compare")
async def compare(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    threshold: float = Query(DEFAULT_THRESHOLD, ge=0.0, le=1.0),
):
    if engines.face is None:
        raise HTTPException(503, "Face models still loading; try again shortly.")
    try:
        emb1, n1 = engines.face.embed_largest(await file1.read())
        emb2, n2 = engines.face.embed_largest(await file2.read())
    except ValueError as e:
        raise HTTPException(400, str(e))

    if emb1 is None:
        raise HTTPException(422, "No face detected in image 1.")
    if emb2 is None:
        raise HTTPException(422, "No face detected in image 2.")

    sim = cosine_similarity(emb1, emb2)
    return {
        "similarity": round(sim, 4),
        "match": is_match(sim, threshold),
        "threshold": threshold,
        "faces_detected": {"image1": n1, "image2": n2},
    }
