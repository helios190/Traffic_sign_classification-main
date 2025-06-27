"""
Bandingkan baseline.h5, model_int8.tflite, model_fp16.onnx
Menampilkan 1 baris per gambar: prediksi dan latensi tiap artefak.

jalankan:
    python -m src.compare_models --ver v2025-06-27 --test-dir test/
"""

from __future__ import annotations
import argparse, pathlib, time, json, sys
import numpy as np, cv2
from .loader import ModelLoader           # loader universal

ROOT = pathlib.Path(__file__).resolve().parents[1]

LABELS = [  # ringkas; lengkapi sesuai 43 kelas
    "Speed 20","Speed 30","Speed 50","Speed 60","Speed 70","Speed 80","End 80",
    "Speed 100","Speed 120","No passing", "No passing 3.5t", "Right-of-way",
    "Priority", "Yield", "Stop", "No vehicles", "No >3.5t", "No entry",
    "Caution", "Curve L", "Curve R", "Double curve", "Bumpy", "Slippery",
    "Narrow", "Road work", "Signals", "Pedestrian", "Children", "Bicycle",
    "Ice", "Animals", "End limit", "Turn R", "Turn L", "Ahead", "Straight/R",
    "Straight/L", "Keep R", "Keep L", "Roundabout", "End nopass",
    "End nopass 3.5t"
]

# ─────────────────────────────────────────────────────────────
def first_name(obj):
    """Ambil elemen pertama, menangani dict/list/tuple."""
    if isinstance(obj, dict):
        return next(iter(obj.values()))
    if isinstance(obj, (list, tuple)):
        return obj[0]
    raise TypeError("Unknown container type")

def tflite_predict_fn(interp):
    """Kembalikan fungsi prediksi untuk interpreter TFLite apa pun."""
    sigs = interp.get_signature_list()

    # — case A: ada SignatureDef —
    if isinstance(sigs, dict) and sigs:
        key      = next(iter(sigs))
        runner   = interp.get_signature_runner(key)
        in_name  = first_name(sigs[key]["inputs"])
        out_name = first_name(sigs[key]["outputs"])
        in_dtype = interp.get_input_details()[0]["dtype"]

        def _fn(arr):
            if in_dtype == np.uint8:
                arr = (arr * 255).astype("uint8")
            return runner(**{in_name: arr})[out_name]
        return _fn

    # — case B: fallback tensor index —
    in_info  = interp.get_input_details()[0]
    out_info = interp.get_output_details()[0]
    in_idx, out_idx = in_info["index"], out_info["index"]
    in_dtype = in_info["dtype"]

    def _fn(arr):
        if in_dtype == np.uint8:
            arr = (arr * 255).astype("uint8")
        interp.set_tensor(in_idx, arr)
        interp.invoke()
        return interp.get_tensor(out_idx)

    return _fn
# ─────────────────────────────────────────────────────────────
def build_predict(model_path: pathlib.Path):
    loader = ModelLoader(model_path)
    if model_path.suffix == ".tflite":
        return tflite_predict_fn(loader.obj)
    return lambda a: loader.predict_logits(a)

def load_imgs(folder: pathlib.Path):
    files = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
    if not files:
        sys.exit(f"[ERR] No images in {folder}")
    imgs, names = [], []
    for p in files:
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (30, 30)).astype("float32") / 255.0
        imgs.append(img)
        names.append(p.name)
    return np.stack(imgs), names
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", default="v2025-06-27")
    ap.add_argument("--test-dir", default="test")
    args = ap.parse_args()

    ver_dir = ROOT / "models" / args.ver
    artefacts = {
        "h5"  : ver_dir / "baseline.h5",
        "int8": ver_dir / "model_int8.tflite",
        "fp16": ver_dir / "model_fp16.onnx",
    }

    # bangun fungsi prediksi utk artefak yg ada
    fns = {k: build_predict(p) for k, p in artefacts.items() if p.exists()}

    imgs, names = load_imgs(ROOT / args.test_dir)
    report = []

    for ix, img in enumerate(imgs):
        row = {"image": names[ix]}
        for key, fn in fns.items():
            t0 = time.time()
            logit = fn(img[None, ...])
            lat = (time.time() - t0) * 1e3
            cls = int(np.argmax(logit))
            row[f"{key}_pred"] = LABELS[cls] if cls < len(LABELS) else str(cls)
            row[f"{key}_ms"] = round(lat, 2)
        report.append(row)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
