"""exp10: Python's end-to-end counterpart to native/gender_predict.cpp —
raw name in, prediction out, via PyTorch. Unlike bench_pytorch.py's --once
mode (which benchmarks the matmul only, on a precomputed dummy input), this
pays real feature-extraction cost every time, for a fair comparison against
the C++ pipeline that also extracts features from the raw string.
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

import torch  # noqa: E402

from tiny_mlp import TinyMLP  # noqa: E402
from gender_features import extract_features  # noqa: E402

in_dim, hidden_dim, out_dim, _n_test = map(int, (artifacts_dir / "shapes.txt").read_text().split())
vocab = (artifacts_dir / "vocab.txt").read_text().splitlines()

model = TinyMLP(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
model.load_state_dict(torch.load(artifacts_dir / "model_state_dict.pt", weights_only=True))
model.eval()

cold_start_ms = (time.perf_counter() - t_start) * 1000.0


def predict(n):
    x = torch.tensor([extract_features(n, vocab)], dtype=torch.float32)
    with torch.no_grad():
        return model(x)[0]


if ONCE:
    predict(name)
else:
    probs = predict(name)
    label = "M" if probs[0] > probs[1] else "F"
    print(json.dumps({"name": name, "pred": label, "prob_M": probs[0].item(), "prob_F": probs[1].item(),
                       "cold_start_ms": cold_start_ms}))
