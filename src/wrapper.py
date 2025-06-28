# src/wrapper.py
from __future__ import annotations
import os
import io
import time
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .loader import ModelLoader            # universal loader
from .traffic import LABELS                # 43 human-readable labels

# ─────────────────── init ──────────────────────────────────────────────
ART = Path(os.environ.get("ARTEFACT_PATH", "Traffic.h5"))
if not ART.exists():
    raise RuntimeError(f"Artefact not found: {ART}")
_loader = ModelLoader(ART)
print(f"[wrapper] loaded {ART.name} via {_loader.backend}")

# ─────────────────── FastAPI app ───────────────────────────────────────
app = FastAPI(title="Traffic-Sign API", version="1.0")

class PredictResponse(BaseModel):
    label: str
    class_id: int
    latency_ms: float

def _preprocess(content: bytes) -> np.ndarray:
    """bytes → NHWC float32[1,30,30,3]"""
    arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Not a valid image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (30, 30)).astype("float32") / 255.0
    return img[None, ...]

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    # read & preprocess
    try:
        data = await file.read()
        x = _preprocess(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # inference + timing
    t0 = time.time()
    logits = _loader.predict_logits(x)
    latency_ms = (time.time() - t0) * 1e3

    # pick class
    cls = int(np.argmax(logits))
    return JSONResponse(
        PredictResponse(
            label=LABELS[cls] if cls < len(LABELS) else str(cls),
            class_id=cls,
            latency_ms=round(latency_ms, 2),
        ).dict()
    )

@app.get("/healthz")
def health():
    return {
        "status": "ok",
        "artefact": ART.name,
        "backend": _loader.backend
    }

# ─────────────────── standalone runner ─────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        "src.wrapper:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
