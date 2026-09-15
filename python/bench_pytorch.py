import json
import pathlib
import sys
import time

ONCE = "--once" in sys.argv  # exp7: one cold invocation, one inference, no printing — see run_cold_invocation_bench.py
_positional = [a for a in sys.argv[1:] if not a.startswith("--")]

t_start = time.perf_counter()

import torch  # noqa: E402  (import timed deliberately — it's part of "cold start" for a Python process)

from tiny_mlp import TinyMLP  # noqa: E402
from bench_common import time_calls, percentiles  # noqa: E402

ARTIFACTS = pathlib.Path(_positional[0]) if _positional else (
    pathlib.Path(__file__).parent.parent / "artifacts"
)

torch.set_num_threads(1)  # match the C++ side: single-threaded, batch size 1

in_dim, hidden_dim, out_dim, _n_test = map(int, (ARTIFACTS / "shapes.txt").read_text().split())

model = TinyMLP(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
model.load_state_dict(torch.load(ARTIFACTS / "model_state_dict.pt", weights_only=True))
model.eval()

x = torch.full((1, in_dim), 0.1)

cold_start_ms = (time.perf_counter() - t_start) * 1000.0


def step():
    with torch.no_grad():
        model(x)


if ONCE:
    step()
else:
    latencies = time_calls(step)
    result = {"impl": "pytorch_eager", "cold_start_ms": cold_start_ms, "warmup_iters": 50, "iters": 500}
    result.update(percentiles(latencies))
    print(json.dumps(result))
