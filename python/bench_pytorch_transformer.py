import json
import pathlib
import sys
import time

t_start = time.perf_counter()

import torch  # noqa: E402

from tiny_transformer import TinyTransformerBlock  # noqa: E402
from bench_common import time_calls, percentiles  # noqa: E402

ONCE = "--once" in sys.argv
_positional = [a for a in sys.argv[1:] if not a.startswith("--")]
ARTIFACTS = pathlib.Path(_positional[0]) if _positional else (
    pathlib.Path(__file__).parent.parent / "artifacts" / "transformer"
)

torch.set_num_threads(1)  # match the C++ side: single-threaded, batch size 1

seq_len, d_model, n_heads, d_ff, _n_test = map(int, (ARTIFACTS / "shapes.txt").read_text().split())

model = TinyTransformerBlock(seq_len, d_model, n_heads, d_ff)
model.load_state_dict(torch.load(ARTIFACTS / "model_state_dict.pt", weights_only=True))
model.eval()

x = torch.full((seq_len, d_model), 0.1)

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
