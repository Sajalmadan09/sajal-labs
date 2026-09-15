import json
import pathlib
import sys
import time

ONCE = "--once" in sys.argv

t_start = time.perf_counter()

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from bench_common import time_calls, percentiles  # noqa: E402

ARTIFACTS = pathlib.Path(__file__).parent.parent / "artifacts" / "bert_tiny"
BENCH_CASE = 1  # matches native/bert.cpp's BENCH_CASE

so = ort.SessionOptions()
so.intra_op_num_threads = 1
session = ort.InferenceSession(str(ARTIFACTS / "model.onnx"), sess_options=so, providers=["CPUExecutionProvider"])

lengths = [int(line.split(" ", 1)[0]) for line in (ARTIFACTS / "test_cases.txt").read_text().splitlines()]
seq_len = lengths[BENCH_CASE]
input_ids = np.fromfile(ARTIFACTS / f"case{BENCH_CASE}_input_ids.bin", dtype=np.int32).reshape(1, seq_len).astype(np.int64)
attention_mask = np.ones((1, seq_len), dtype=np.int64)
token_type_ids = np.zeros((1, seq_len), dtype=np.int64)

cold_start_ms = (time.perf_counter() - t_start) * 1000.0


def step():
    session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids})


if ONCE:
    step()
else:
    latencies = time_calls(step)
    result = {"impl": "onnxruntime_cpu", "cold_start_ms": cold_start_ms, "warmup_iters": 50, "iters": 500}
    result.update(percentiles(latencies))
    print(json.dumps(result))
