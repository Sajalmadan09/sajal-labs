import json
import pathlib
import time

t_start = time.perf_counter()

import torch  # noqa: E402

from tiny_transformer import TinyTransformerBlock, SEQ_LEN, D_MODEL  # noqa: E402
from bench_common import time_calls, percentiles  # noqa: E402

ARTIFACTS = pathlib.Path(__file__).parent.parent / "artifacts" / "transformer"

torch.set_num_threads(1)  # match the C++ side: single-threaded, batch size 1

model = TinyTransformerBlock()
model.load_state_dict(torch.load(ARTIFACTS / "model_state_dict.pt", weights_only=True))
model.eval()

x = torch.full((SEQ_LEN, D_MODEL), 0.1)

cold_start_ms = (time.perf_counter() - t_start) * 1000.0


def step():
    with torch.no_grad():
        model(x)


latencies = time_calls(step)
result = {"impl": "pytorch_eager", "cold_start_ms": cold_start_ms, "warmup_iters": 50, "iters": 500}
result.update(percentiles(latencies))
print(json.dumps(result))
