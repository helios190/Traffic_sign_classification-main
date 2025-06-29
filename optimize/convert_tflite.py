import pathlib
import random
import sys

import cv2
import numpy as np
import tensorflow as tf

VER = "v2025-06-27"
SM_DIR = pathlib.Path(f"models/{VER}/savedmodel")
OUT_PATH = pathlib.Path(f"models/{VER}/model_int8.tflite")

# ----------------------------------------------------------------------
# 1. Validasi input
# ----------------------------------------------------------------------
if not SM_DIR.exists():
    sys.exit(f"[ERR] SavedModel tidak ditemukan: {SM_DIR}")

SAMPLES = list(pathlib.Path("test").glob("*.png"))
REP_N = min(20, len(SAMPLES))     # ganti 20 sesuai aset Anda

def representative_dataset():
    for path in random.sample(SAMPLES, REP_N):
        img = cv2.imread(str(path))          # BGR
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (30, 30)).astype("float32") / 255.0
        yield [img[None, ...]]

# ----------------------------------------------------------------------
# 3. Konversi TFLite INT8
# ----------------------------------------------------------------------
converter = tf.lite.TFLiteConverter.from_saved_model(str(SM_DIR))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

OUT_PATH.write_bytes(converter.convert())
print("✅  model_int8.tflite sukses ditulis →", OUT_PATH)


