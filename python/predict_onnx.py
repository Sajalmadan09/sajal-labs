"""exp10: Python's end-to-end counterpart to native/gender_predict.cpp, via
ONNX Runtime — see predict_pytorch.py's docstring for why this pays real
feature-extraction cost rather than benchmarking the matmul in isolation.
"""
import json
import pathlib
import sys
import time

ONCE = "--once" in sys.argv
_positional = [a for a in sys.argv[1:] if not a.startswith("--")]
artifacts_dir = pathlib.Path(_positional[0])
name = _positional[1] if len(_positional) > 1 else "sajal"

t_start = time.perf_counter()

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from gender_features import extract_features  # noqa: E402

vocab = (artifacts_dir / "vocab.txt").read_text().splitlines()

so = ort.SessionOptions()
so.intra_op_num_threads = 1
session = ort.InferenceSession(
    str(artifacts_dir / "model.onnx"), sess_options=so, providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name

cold_start_ms = (time.perf_counter() - t_start) * 1000.0


def predict(n):
    x = np.array([extract_features(n, vocab)], dtype=np.float32)
    return session.run(None, {input_name: x})[0][0]


if ONCE:
    predict(name)
else:
    probs = predict(name)
    label = "M" if probs[0] > probs[1] else "F"
    print(json.dumps({"name": name, "pred": label, "prob_M": float(probs[0]), "prob_F": float(probs[1]),
                       "cold_start_ms": cold_start_ms}))
