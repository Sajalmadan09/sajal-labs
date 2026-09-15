import json
import pathlib
import sys
import time

ONCE = "--once" in sys.argv

t_start = time.perf_counter()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import BertModel  # noqa: E402

from bench_common import time_calls, percentiles  # noqa: E402

ARTIFACTS = pathlib.Path(__file__).parent.parent / "artifacts" / "bert_tiny"
BENCH_CASE = 1  # matches native/bert.cpp's BENCH_CASE — same 12-token sentence, fair comparison

torch.set_num_threads(1)

# Loads from the local HF cache (already downloaded by bert_export.py) — no
# network call, matching how a real deployment behaves after first download.
model = BertModel.from_pretrained("prajjwal1/bert-tiny")
model.eval()

lengths = [int(line.split(" ", 1)[0]) for line in (ARTIFACTS / "test_cases.txt").read_text().splitlines()]
input_ids = np.fromfile(ARTIFACTS / f"case{BENCH_CASE}_input_ids.bin", dtype=np.int32).reshape(1, lengths[BENCH_CASE])
input_ids = torch.tensor(input_ids, dtype=torch.long)

cold_start_ms = (time.perf_counter() - t_start) * 1000.0


def step():
    with torch.no_grad():
        model(input_ids)


if ONCE:
    step()
else:
    latencies = time_calls(step)
    result = {"impl": "pytorch_eager", "cold_start_ms": cold_start_ms, "warmup_iters": 50, "iters": 500}
    result.update(percentiles(latencies))
    print(json.dumps(result))
