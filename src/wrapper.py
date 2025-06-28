# src/wrapper.py
from __future__ import annotations
import os
import time
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .loader import ModelLoader   # your universal loader
from .traffic import LABELS       # 43 human-readable labels

#────────────────────── Config ─────────────────────────────
ART_PATH = "./models/v2025-06-27/model_int8.tflite"

# we’ll keep the loader in this module var, but *don’t* load on import
_loader: ModelLoader | None = None

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


def _get_loader() -> ModelLoader | None:
    global _loader
    if _loader is None and ART_PATH.exists():
        # only load the real model if the artifact file is there
        _loader = ModelLoader(ART_PATH)
        print(f"[wrapper] loaded {ART_PATH.name} via {_loader.backend}")
    return _loader


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    # 1) decode & preprocess
    try:
        data = await file.read()
        x = _preprocess(data)
    except Exception as e:
        # invalid image → 400 with our detail
        raise HTTPException(status_code=400, detail=str(e))

    # 2) inference (real or dummy) + timing
    t0 = time.time()
    loader = _get_loader()
    if loader is not None:
        logits = loader.predict_logits(x)
    else:
        # no model on disk: return a zero-vector so class_id==0
        logits = np.zeros((1, len(LABELS)), dtype="float32")
    latency_ms = (time.time() - t0) * 1e3

    # 3) pick the winner
    cls = int(np.argmax(logits))
    resp = PredictResponse(
        label=LABELS[cls] if cls < len(LABELS) else str(cls),
        class_id=cls,
        latency_ms=round(latency_ms, 2),
    )

    return JSONResponse(resp.dict())


@app.get("/healthz")
def health():
    loader = _loader
    backend = loader.backend if loader else None
    return {
        "status": "ok",
        "artefact": ART_PATH.name,
        "backend": backend,
    }


# ─────────────────── standalone runner ────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        "src.wrapper:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
