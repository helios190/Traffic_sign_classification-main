# src/wrapper.py
from fastapi import FastAPI
import UploadFile
import File
import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import time
import numpy as np
import cv2
from pathlib import Path
from .loader import ModelLoader
from .traffic import LABELS

ART_PATH = Path(os.environ.get("ARTEFACT_PATH", "Traffic.h5"))

_loader: ModelLoader  # will be set in startup

app = FastAPI(title="Traffic-Sign API", version="1.0")

@app.on_event("startup")
def load_model():
    if not ART_PATH.exists():
        raise RuntimeError(f"Artefact not found: {ART_PATH}")
    global _loader
    _loader = ModelLoader(ART_PATH)
    print(f"[wrapper] loaded {ART_PATH.name} via {_loader.backend}")

class PredictResponse(BaseModel):
    label: str
    class_id: int
    latency_ms: float

def _preprocess(content: bytes) -> np.ndarray:
    arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Not a valid image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (30, 30)).astype("float32") / 255.0
    return img[None, ...]

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        x = _preprocess(data)
    except Exception as e:
        raise HTTPException(400, str(e))

    t0 = time.time()
    logits = _loader.predict_logits(x)
    latency = (time.time() - t0)*1e3

    cls = int(np.argmax(logits))
    return JSONResponse(PredictResponse(
        label=LABELS[cls] if cls < len(LABELS) else str(cls),
        class_id=cls,
        latency_ms=round(latency,2)
    ).dict())

@app.get("/healthz")
def health():
    return {"status":"ok","artefact":ART_PATH.name,"backend":_loader.backend}
