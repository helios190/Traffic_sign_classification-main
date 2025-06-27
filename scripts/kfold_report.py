import numpy as np, pandas as pd, cv2, tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from src.model import load, preprocess
from src.data_utils import load_meta

df = load_meta("dataset/train")
X = np.stack([preprocess(p)[0] for p in df["path"]])
y = df["label"].astype(int).to_numpy()

kf = StratifiedKFold(5, shuffle=True, random_state=42)
scores = []
for tr, va in kf.split(X, y):
    mdl = load()
    preds = np.argmax(mdl.predict(X[va]), axis=1)
    scores.append(accuracy_score(y[va], preds))
print(f"CV accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
