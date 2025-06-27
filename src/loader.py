from pathlib import Path
import numpy as np, tensorflow as tf

try: import onnxruntime as ort
except ImportError: ort=None
try: import tensorrt as trt, pycuda.driver as cuda, pycuda.autoinit
except ImportError: trt=None

class ModelLoader:
    def __init__(self, artefact: str | Path):
        self.path = Path(artefact)
        self.backend, self.obj = self._load()

    # ─────────────────────────────
    def _load(self):
        if self.path.suffix == ".h5":
            return "keras", tf.keras.models.load_model(self.path, compile=False)
        if self.path.is_dir():
            return "saved", tf.saved_model.load(str(self.path))
        if self.path.suffix == ".tflite":
            itp = tf.lite.Interpreter(model_path=str(self.path)); itp.allocate_tensors()
            return "tflite", itp
        if self.path.suffix == ".onnx":
            assert ort, "pip install onnxruntime"
            return "onnx", ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        if self.path.suffix == ".plan":
            assert trt, "install TensorRT & pycuda"
            return "trt", self._load_trt()
        raise ValueError("format tidak didukung")

    def _load_trt(self):
        engine = trt.Runtime(trt.Logger.WARNING).deserialize_cuda_engine(self.path.read_bytes())
        ctx = engine.create_execution_context()
        d_in  = cuda.mem_alloc(trt.volume(engine.get_binding_shape(0))*4)
        d_out = cuda.mem_alloc(trt.volume(engine.get_binding_shape(1))*4)
        stream = cuda.Stream()
        return (engine, ctx, d_in, d_out, stream)

    # ─────────────────────────────
    def predict_logits(self, arr: np.ndarray):
        if self.backend=="keras":
            return self.obj.predict(arr, verbose=0)
        if self.backend=="saved":
            fn=self.obj.signatures["serving_default"]; return fn(tf.constant(arr))["output_0"].numpy()
        if self.backend=="tflite":
            runner=self.obj.get_signature_runner(); return runner(features=arr)["output_0"]
        if self.backend=="onnx":
            return self.obj.run(None, {"input": arr.astype("float32")})[0]
        if self.backend=="trt":
            return self._run_trt(arr)
        raise RuntimeError("backend unknown")

    def _run_trt(self, arr: np.ndarray):
        engine, ctx, d_in, d_out, stream = self.obj
        import pycuda.driver as cuda, numpy as np
        h_in = arr.astype(np.float32).ravel()
        h_out = np.empty(ctx.get_binding_shape(1), dtype=np.float32)
        cuda.memcpy_htod_async(d_in, h_in, stream)
        ctx.execute_async_v2([int(d_in), int(d_out)], stream.handle)
        cuda.memcpy_dtoh_async(h_out, d_out, stream)
        stream.synchronize()
        return h_out.reshape(1, -1)
