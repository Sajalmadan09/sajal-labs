"""exp14's ONNX Runtime end-to-end counterpart — ONNX Runtime doesn't do
tokenization itself, so this uses BertTokenizerFast (the same real
tokenizer as the PyTorch path) for preprocessing, then ONNX Runtime for
the encoder. Realistic: ONNX deployments commonly still pair with a
standalone/HF tokenizer for preprocessing."""
import sys
import time

ONCE = "--once" in sys.argv

t_start = time.perf_counter()

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from transformers import BertTokenizerFast  # noqa: E402

import pathlib
ARTIFACTS = pathlib.Path(__file__).parent.parent / "artifacts" / "bert_tiny"

tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")
so = ort.SessionOptions()
so.intra_op_num_threads = 1
session = ort.InferenceSession(str(ARTIFACTS / "model.onnx"), sess_options=so, providers=["CPUExecutionProvider"])

cold_start_ms = (time.perf_counter() - t_start) * 1000.0

text = "The quick brown fox jumps over the lazy dog."
encoded = tokenizer(text, return_tensors="np")
outputs = session.run(None, {"input_ids": encoded["input_ids"].astype(np.int64),
                              "attention_mask": encoded["attention_mask"].astype(np.int64),
                              "token_type_ids": encoded["token_type_ids"].astype(np.int64)})

if not ONCE:
    print(f"cold_start_ms={cold_start_ms:.2f}  pooled[:5]={outputs[1][0][:5].tolist()}")
