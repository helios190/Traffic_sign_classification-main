# tests/test_model_pipeline.py
import numpy as np
import pytest
from pathlib import Path

from src.loader import ModelLoader

# Path ke artefak model di repo kamu
ROOT = Path(__file__).resolve().parents[1]
H5_PATH     = ROOT / "models" / "v2025-06-27" / "baseline.h5"
TFLITE_PATH = ROOT / "models" / "v2025-06-27" / "model_int8.tflite"

@pytest.fixture
def dummy_input():
    # batch=1, H×W×C = 30×30×3, float32 in [0,1]
    return np.random.rand(1, 30, 30, 3).astype("float32")

@pytest.mark.parametrize("model_path", [H5_PATH, TFLITE_PATH])
def test_output_shape_and_type(model_path, dummy_input):
    loader = ModelLoader(model_path)
    logits = loader.predict_logits(dummy_input)
    # harus (1,43) dan float-like
    assert logits.shape == (1, 43), f"{model_path.name} returned wrong shape"
    assert issubclass(logits.dtype.type, np.floating)

def test_h5_vs_tflite_numerical_consistency(dummy_input):
    out_h5     = ModelLoader(H5_PATH).predict_logits(dummy_input)
    out_tflite = ModelLoader(TFLITE_PATH).predict_logits(dummy_input)

    # Hitung selisih absolut terbesar
    diffs = np.abs(out_h5 - out_tflite)
    max_diff = float(np.max(diffs))
    # Toleransi ~0.1–0.2 biasanya wajar untuk quantized INT8
    assert max_diff < 0.2, f"Diff too large: {max_diff:.3f}"
