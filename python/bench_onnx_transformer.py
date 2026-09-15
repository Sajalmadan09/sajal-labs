import json
import pathlib
import sys
import time

t_start = time.perf_counter()

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from bench_common import time_calls, percentiles  # noqa: E402

ARTIFACTS = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (
    pathlib.Path(__file__).parent.parent / "artifacts" / "transformer"
)

seq_len, d_model, _n_heads, _d_ff, _n_test = map(int, (ARTIFACTS / "shapes.txt").read_text().split())

so = ort.SessionOptions()
so.intra_op_num_threads = 1  # match the C++ side: single-threaded, batch size 1
session = ort.InferenceSession(
    str(ARTIFACTS / "model.onnx"), sess_options=so, providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name

x = np.full((seq_len, d_model), 0.1, dtype=np.float32)

cold_start_ms = (time.perf_counter() - t_start) * 1000.0


def step():
    session.run(None, {input_name: x})


latencies = time_calls(step)
result = {"impl": "onnxruntime_cpu", "cold_start_ms": cold_start_ms, "warmup_iters": 50, "iters": 500}
result.update(percentiles(latencies))
print(json.dumps(result))
