from __future__ import annotations
import os, io, time
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .loader import ModelLoader            # <— your universal loader
from .traffic import LABELS                # mapping 0-42 ➜ human label

# ─────────────────── init ──────────────────────────────────────────────
ARTEFACT = "./models/v2025-06-27/model_int8.tflite"

_loader = ModelLoader(ARTEFACT)            # load once at import
print(f"[wrapper] loaded {ARTEFACT} via {_loader.backend}")

# ─────────────────── FastAPI app ───────────────────────────────────────
app = FastAPI(title="Traffic-Sign API", version="1.0")

class PredictResponse(BaseModel):
    label: str
    class_id: int
    latency_ms: float

def _preprocess(content: bytes) -> np.ndarray:
    """bytes (image) ➜ NHWC float32[1,30,30,3]"""
    arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Not a valid image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (30, 30)).astype("float32") / 255.0
    return img[None, ...]                  # add batch dim

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        x = _preprocess(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    t0 = time.time()
    logits = _loader.predict_logits(x)
    latency = (time.time() - t0) * 1e3

    cls = int(np.argmax(logits))
    payload = PredictResponse(
        label=LABELS[cls] if cls < len(LABELS) else str(cls),
        class_id=cls,
        latency_ms=round(latency, 2),
    ).dict()          # ← ganti model_dump() → dict()

    return JSONResponse(payload)
# Optional health-check
@app.get("/healthz")
def health():
    return {"status": "ok", "model": ARTEFACT.name, "backend": _loader.backend}
