from pathlib import Path
import numpy as np, tensorflow as tf

try: import onnxruntime as ort
except ImportError: ort=None


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

    # ─────────────────────────────
    def predict_logits(self, arr: np.ndarray):
        if self.backend=="keras":
            return self.obj.predict(arr, verbose=0)
        if self.backend=="saved":
            fn=self.obj.signatures["serving_default"]; return fn(tf.constant(arr))["output_0"].numpy()
# ─ inside predict_logits() ───────────────────────────────────────────
        if self.backend == "tflite":
            interp   = self.obj
            sig_map  = interp.get_signature_list()

            def _maybe_quantize(a: np.ndarray, dtype):
                """Jika interpreter butuh uint8 → ubah 0-1 float ke 0-255 uint8."""
                if dtype == np.uint8:
                    return (a * 255).astype("uint8")
                return a.astype("float32")

            def _first_name(container):
                """Return first tensor name from dict | list | tuple."""
                if isinstance(container, dict):
                    return next(iter(container.values()))
                if isinstance(container, (list, tuple)):
                    return container[0]
                raise TypeError("Unknown Signature I/O type")

            if isinstance(sig_map, dict) and sig_map:
                sig_key  = next(iter(sig_map))
                runner   = interp.get_signature_runner(sig_key)
                in_name  = _first_name(sig_map[sig_key]["inputs"])
                out_name = _first_name(sig_map[sig_key]["outputs"])
                in_dtype = interp.get_input_details()[0]["dtype"]
                return runner(**{in_name: _maybe_quantize(arr, in_dtype)})[out_name]
            else:
                # fallback pakai tensor-index
                in_info = interp.get_input_details()[0]
                out_idx = interp.get_output_details()[0]["index"]
                interp.set_tensor(
                    in_info["index"], _maybe_quantize(arr, in_info["dtype"])
                )
                interp.invoke()
                return interp.get_tensor(out_idx)
        raise RuntimeError("backend unknown")
