# optimize/export_savedmodel.py
import tensorflow as tf
from pathlib import Path
import sys

# --- ubah satu-satunya variabel ini saat merilis model baru -----------
VER = "v2025-06-27"                         # contoh: tanggal atau tag git
# ----------------------------------------------------------------------

root = Path("models") / VER
src  = root / "baseline.h5"
dst  = root / "savedmodel"

# 1) Pastikan checkpoint .h5 ada
if not src.exists():
    sys.exit(f"[ERR] tidak menemukan file: {src}")

# 2) Buat folder tujuan
dst.mkdir(parents=True, exist_ok=True)

# 3) Konversi
model = tf.keras.models.load_model(src, compile=False)
model.save(dst)

print(f"✅ SavedModel berhasil dibuat → {dst}")
