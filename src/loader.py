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
            import tensorflow as tf
        self.backend, self.obj = self._load()

    def _load(self):
        suffix = self.path.suffix.lower()

        if suffix == ".h5":
            # keras H5
            import tensorflow as tf
            model = tf.keras.models.load_model(self.path, compile=False)
            return "keras", model

        if self.path.is_dir():
            # saved_model dir
            import tensorflow as tf
            saved = tf.saved_model.load(str(self.path))
            return "saved", saved

        if suffix == ".tflite":
            # TFLite file
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
            return self.obj.predict(arr, verbose=0)

        if self.backend == "saved":
            import tensorflow as tf
            fn = self.obj.signatures["serving_default"]
            return fn(tf.constant(arr))["output_0"].numpy()

        if self.backend == "tflite":
            interp = self.obj
            sig_map = interp.get_signature_list()
            def _maybe_quantize(a, dtype):
                return (a * 255).astype("uint8") if dtype == np.uint8 else a.astype("float32")

            def _first_name(c):
                if isinstance(c, dict):
                    return next(iter(c.values()))
                if isinstance(c, (list, tuple)):
                    return c[0]
                raise TypeError

            if isinstance(sig_map, dict) and sig_map:
                key = next(iter(sig_map))
                runner = interp.get_signature_runner(key)
                in_name = _first_name(sig_map[key]["inputs"])
                out_name = _first_name(sig_map[key]["outputs"])
                in_dtype = interp.get_input_details()[0]["dtype"]
                return runner(**{in_name: _maybe_quantize(arr, in_dtype)})[out_name]
            else:
                # fallback
                inp = interp.get_input_details()[0]
                out = interp.get_output_details()[0]["index"]
                interp.set_tensor(inp["index"], _maybe_quantize(arr, inp["dtype"]))
                interp.invoke()
                return interp.get_tensor(out)

        raise RuntimeError(f"backend unknown: {self.backend!r}")
