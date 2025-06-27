from pathlib import Path
from typing import List, Dict

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------------
# 1. Muat model sekali saja (singleton)
# ---------------------------------------------------------------------
_MODEL = None
def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = tf.keras.models.load_model("Traffic.h5", compile=False)
    return _MODEL

# ---------------------------------------------------------------------
# 2. Label list (index 0-42)
# ---------------------------------------------------------------------
LABELS: List[str] = [
    "Speed limit (20km/h)",                # 0
    "Speed limit (30km/h)",                # 1
    "Speed limit (50km/h)",                # 2
    "Speed limit (60km/h)",                # 3
    "Speed limit (70km/h)",                # 4
    "Speed limit (80km/h)",                # 5
    "End of speed limit (80km/h)",         # 6
    "Speed limit (100km/h)",               # 7
    "Speed limit (120km/h)",               # 8
    "No passing",                          # 9
    "No passing for vehicles > 3.5 t",     # 10
    "Right-of-way at intersection",        # 11
    "Priority road",                       # 12
    "Yield",                               # 13
    "Stop",                                # 14
    "No vehicles",                         # 15
    "Vehicles > 3.5 t prohibited",         # 16
    "No entry",                            # 17
    "General caution",                     # 18
    "Dangerous curve left",                # 19
    "Dangerous curve right",               # 20
    "Double curve",                        # 21
    "Bumpy road",                          # 22
    "Slippery road",                       # 23
    "Road narrows on the right",           # 24
    "Road work",                           # 25
    "Traffic signals",                     # 26
    "Pedestrians",                         # 27
    "Children crossing",                   # 28
    "Bicycles crossing",                   # 29
    "Beware of ice/snow",                  # 30
    "Wild animals crossing",               # 31
    "End of all speed & passing limits",   # 32
    "Turn right ahead",                    # 33
    "Turn left ahead",                     # 34
    "Ahead only",                          # 35
    "Go straight or right",                # 36
    "Go straight or left",                 # 37
    "Keep right",                          # 38
    "Keep left",                           # 39
    "Roundabout mandatory",                # 40
    "End of no passing",                   # 41
    "End of no passing > 3.5 t",           # 42
]

# ---------------------------------------------------------------------
# 3. Kelas wrapper – kompatibel dgn versi Anda
# ---------------------------------------------------------------------
class Traffic:
    def __init__(self, filename: str | Path):
        self.filename = str(filename)

    def trafficsign(self) -> List[Dict[str, str]]:
        """Return [{"image": <predicted_label>}] or [{"ERROR": …}]"""
        try:
            # -------- load & preprocess image --------
            img_bgr = cv2.imread(self.filename)
            if img_bgr is None:
                raise FileNotFoundError(self.filename)

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb).resize((30, 30))
            arr     = (np.asarray(img_pil, dtype="float32") / 255.0)[None, ...]

            # -------- inference --------
            model   = _get_model()
            pred_id = int(np.argmax(model.predict(arr, verbose=0), axis=1)[0])

            # -------- map to label --------
            label = LABELS[pred_id]
            return [{"image": label}]

        except Exception as e:
            return [{"ERROR": f"{e}"}]
