from pathlib import Path
import numpy as np
try:
    from tflite_runtime.interpreter import Interpreter
    TF_LITE = True
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    TF_LITE = False


class ModelLoader:
    def __init__(self, artefact: str | Path):
        self.path = Path(artefact)
        self.backend, self.obj = self._load()

    def _load(self):
        suffix = self.path.suffix.lower()
        if suffix == ".h5":
            # Keras H5
            import tensorflow as tf
            return "keras", tf.keras.models.load_model(self.path, compile=False)

        if self.path.is_dir():
            # TF SavedModel
            import tensorflow as tf
            return "saved", tf.saved_model.load(str(self.path))

        if suffix == ".tflite":
            itp = Interpreter(model_path=str(self.path))
            itp.allocate_tensors()
            return "tflite", itp

        raise RuntimeError(f"Unsupported artefact: {self.path}")

    def predict_logits(self, arr: np.ndarray):
        if self.backend == "keras":
            return self.obj.predict(arr, verbose=0)

        if self.backend == "saved":
            import tensorflow as tf
            fn = self.obj.signatures["serving_default"]
            return fn(tf.constant(arr))["output_0"].numpy()

        if self.backend == "tflite":
            interp: Interpreter = self.obj

            # helper to quantize input if required
            def _maybe_quantize(a: np.ndarray, dtype):
                if dtype == np.uint8 or dtype == np.int8:
                    # scale 0–1→0–255 or -128→127
                    info = interp.get_input_details()[0]
                    zp = info["quantization_parameters"]["zero_points"][0]
                    scale = info["quantization_parameters"]["scales"][0]
                    # real = (quant - zp)*scale  → quant = real/scale + zp
                    q = np.round(a / scale + zp)
                    qtype = np.uint8 if info["dtype"] == np.uint8 else np.int8
                    return np.clip(q, np.iinfo(qtype).min, np.iinfo(qtype).max).astype(qtype)
                return a.astype("float32")

            # find input & output details
            inp = interp.get_input_details()[0]
            out = interp.get_output_details()[0]

            # set tensor & invoke
            quantized = _maybe_quantize(arr, inp["dtype"])
            interp.set_tensor(inp["index"], quantized)
            interp.invoke()
            raw = interp.get_tensor(out["index"])

            # dequantize if needed
            zp = out["quantization_parameters"]["zero_points"][0]
            scale = out["quantization_parameters"]["scales"][0]
            if out["dtype"] in (np.uint8, np.int8):
                return (raw.astype("float32") - zp) * scale

            return raw.astype("float32")

        raise RuntimeError("Unknown backend")
