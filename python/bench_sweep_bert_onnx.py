"""exp12's ONNX warm-loop bench, generalized to any bert_sweep config dir
(unlike bench_bert_onnx.py, which is pinned to exp11's real bert-tiny
artifacts) — same warmup+500-iteration methodology as every other bench_*
script in this project.
"""
import json
import pathlib
import sys

import numpy as np
import onnxruntime as ort

from bench_common import time_calls, percentiles

ARTIFACTS = pathlib.Path(sys.argv[1])
SEQ_LEN = int((ARTIFACTS / "test_cases.txt").read_text().splitlines()[1].split()[0])

so = ort.SessionOptions()
so.intra_op_num_threads = 1
session = ort.InferenceSession(str(ARTIFACTS / "model.onnx"), sess_options=so, providers=["CPUExecutionProvider"])

input_ids = np.fromfile(ARTIFACTS / "case1_input_ids.bin", dtype=np.int32).reshape(1, SEQ_LEN).astype(np.int64)
attention_mask = np.ones((1, SEQ_LEN), dtype=np.int64)
token_type_ids = np.zeros((1, SEQ_LEN), dtype=np.int64)


def step():
    session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids})


latencies = time_calls(step)
result = {"impl": "onnxruntime_cpu"}
result.update(percentiles(latencies))
print(json.dumps(result))
