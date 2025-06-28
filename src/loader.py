from pathlib import Path
import numpy as np

# Attempt to use the lightweight TFLite runtime; fall back to full TF if needed
try:
    from tflite_runtime.interpreter import Interpreter
    TF_LITE = True
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    TF_LITE = False


class ModelLoader:
    """
    Universal loader for Keras (.h5), SavedModel dir, or TFLite (.tflite).
    """
    def __init__(self, artefact: str | Path):
        self.path = Path(artefact)
        self.backend, self.obj = self._load()

    def _load(self):
        # Keras model (.h5)
        if self.path.suffix == ".h5":
            # full TF must be available
            return "keras", tf.keras.models.load_model(self.path, compile=False)

        # TensorFlow SavedModel directory
        if self.path.is_dir():
            return "saved", tf.saved_model.load(str(self.path))

        # TFLite flatbuffer (.tflite)
        if self.path.suffix == ".tflite":
            interpreter = Interpreter(model_path=str(self.path))
            interpreter.allocate_tensors()
            return "tflite", interpreter

        raise RuntimeError(f"Unsupported artefact type: {self.path}")

    def predict_logits(self, arr: np.ndarray) -> np.ndarray:
        # Keras inference
        if self.backend == "keras":
            return self.obj.predict(arr, verbose=0)

        # SavedModel inference via signature
        if self.backend == "saved":
            fn = self.obj.signatures.get("serving_default")
            if fn is None:
                raise RuntimeError("SavedModel missing 'serving_default' signature")
            out = fn(tf.constant(arr))
            # assume output key is 'output_0' or first
            key = next(iter(out))
            return out[key].numpy()

        # TFLite inference
        if self.backend == "tflite":
            interp = self.obj
            details_in = interp.get_input_details()[0]
            details_out = interp.get_output_details()[0]

            # handle uint8 quantized models
            data = arr.astype("float32")
            if details_in["dtype"] == np.uint8:
                data = (data * 255).astype(np.uint8)

            interp.set_tensor(details_in["index"], data)
            interp.invoke()
            return interp.get_tensor(details_out["index"])

        raise RuntimeError(f"Unknown backend: {self.backend}")
