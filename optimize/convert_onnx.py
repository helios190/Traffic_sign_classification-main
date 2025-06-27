# optimize/convert_onnx.py
import subprocess, pathlib, sys

VER = "v2025-06-27"
sm_dir   = f"models/{VER}/savedmodel"
onnx_fp32 = f"models/{VER}/model_fp32.onnx"
onnx_fp16 = f"models/{VER}/model_fp16.onnx"

# ── 1) SavedModel ➜ ONNX FP32 ───────────────────────────
subprocess.check_call([
    sys.executable, "-m", "tf2onnx.convert",
    "--saved-model", sm_dir,
    "--output",      onnx_fp32,
    "--opset",       "18"
])

# ── 2) ONNX FP32 ➜ FP16  ────────────────────────────────
subprocess.check_call([
    sys.executable, "-m", "onnxruntime.tools.convert_float_to_float16",
    "--input",  onnx_fp32,
    "--output", onnx_fp16
])

print("✅  ONNX FP16 siap  →", onnx_fp16)
