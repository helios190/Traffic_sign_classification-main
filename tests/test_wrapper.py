import io, numpy as np, cv2
from fastapi.testclient import TestClient
from src.wrapper import app               # ← import the FastAPI app

cli = TestClient(app)


def _dummy_png() -> bytes:
    """Return a 30×30 blue square encoded as PNG bytes."""
    arr = np.zeros((30, 30, 3), dtype="uint8")
    arr[:] = (255, 0, 0)                  # BGR blue
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return buf.tobytes()


def test_predict_ok():
    resp = cli.post(
        "/predict",
        files={"file": ("blue.png", _dummy_png(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"label", "class_id", "latency_ms"}
    assert isinstance(body["class_id"], int)
    assert body["latency_ms"] > 0


def test_predict_bad_file():
    resp = cli.post(
        "/predict",
        files={"file": ("junk.txt", b"not-an-image", "text/plain")},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Not a valid image"
