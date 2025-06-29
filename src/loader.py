# src/loader.py
from pathlib import Path
import numpy as np
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    TF_LITE = True
except ImportError:
    TF_LITE = False

class ModelLoader:
    def __init__(self, artefact: str | Path):
        self.path = Path(artefact)
        if not TF_LITE:
            import tensorflow as tf  # for CPU fallback interpreter
        self.backend, self.obj = self._load()

    def _load(self):
        suffix = self.path.suffix.lower()

        if suffix == ".h5":
            import tensorflow as tf
            model = tf.keras.models.load_model(self.path, compile=False)
            return "keras", model

        if self.path.is_dir():
            import tensorflow as tf
            saved = tf.saved_model.load(str(self.path))
            return "saved", saved

        if suffix == ".tflite":
            if TF_LITE:
                itp = TFLiteInterpreter(model_path=str(self.path))
            else:
                import tensorflow as tf
                itp = tf.lite.Interpreter(model_path=str(self.path))
            itp.allocate_tensors()
            return "tflite", itp

        raise RuntimeError(f"Unknown artefact type: {self.path!r}")

    def predict_logits(self, arr: np.ndarray):
        if self.backend == "keras":
            # H5 model: returns float32 logits directly
            return self.obj.predict(arr, verbose=0)

        if self.backend == "tflite":
            interp = self.obj
            # 1) Retrieve tensor details
            in_detail = interp.get_input_details()[0]
            out_detail = interp.get_output_details()[0]

            # 2) Quantization parameters
            scale_in, zp_in = in_detail.get('quantization', (0.0, 0))
            dtype_in = in_detail['dtype']
            scale_out, zp_out = out_detail.get('quantization', (0.0, 0))

            # 3) Quantize input if model expects integers
            if scale_in and zp_in and np.issubdtype(dtype_in, np.integer):
                arr_q = np.clip(
                    np.round(arr / scale_in) + zp_in,
                    np.iinfo(dtype_in).min,
                    np.iinfo(dtype_in).max
                ).astype(dtype_in)
            else:
                arr_q = arr.astype(dtype_in)

            # 4) Run inference
            interp.set_tensor(in_detail['index'], arr_q)
            interp.invoke()

            # 5) Get raw output and dequantize if needed
            out_q = interp.get_tensor(out_detail['index'])
            if scale_out and zp_out and np.issubdtype(out_q.dtype, np.integer):
                out = (out_q.astype(np.float32) - zp_out) * scale_out
            else:
                out = out_q.astype(np.float32)

            return out

        if self.backend == "saved":
            # SavedModel: use default signature
            import tensorflow as tf
            fn = self.obj.signatures['serving_default']
            # TensorFlow SavedModel signature outputs tensors
            result = fn(tf.constant(arr))
            # Assume single output
            return next(iter(result.values())).numpy()

        raise RuntimeError(f"Unknown backend: {self.backend!r}")