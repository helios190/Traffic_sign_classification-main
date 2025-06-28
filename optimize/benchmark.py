import time
import numpy as np
import json
import pathlib
import tensorflow as tf
import onnxruntime as ort
from src.loader import ModelLoader
IMG = np.random.rand(1,30,30,3).astype("float32")

artifacts = [
    "baseline.h5","savedmodel","model_int8.tflite",
    "model_fp16.onnx","model_fp16.plan"
]
rows=[]
for art in artifacts:
    p = pathlib.Path("models/v2025-06-27")/art
    if not p.exists(): continue
    loader = ModelLoader(p)
    t0=time.time()
    for _ in range(100):
        _ = loader.predict_logits(IMG)
    lat_ms = (time.time()-t0)/100*1e3
    rows.append({"name":p.name,"lat_ms":round(lat_ms,2),"size_mb":round(p.stat().st_size/1e6,2)})
print(json.dumps(rows,indent=2))
