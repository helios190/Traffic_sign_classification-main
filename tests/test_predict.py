from src import traffic
import numpy as np

def test_forward_dummy():
    dummy = np.random.rand(1, 30, 30, 3).astype("float32")
    mdl = traffic.load()
    assert mdl.predict(dummy).shape == (1, 43)