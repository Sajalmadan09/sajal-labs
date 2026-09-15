"""exp14: the properly minimal ONNX Runtime deployment — `tokenizers`
(HuggingFace's standalone Rust tokenizer library) instead of full
`transformers` for preprocessing. predict_bert_onnx.py used
BertTokenizerFast (from transformers) for convenience and paid ~2.5s of
import cost for it; this is what a deployment optimizing for cold-start
would actually do instead, and produces byte-identical token IDs (verified
against the same reference)."""
import pathlib
import sys
import time

ONCE = "--once" in sys.argv

t_start = time.perf_counter()

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from tokenizers import BertWordPieceTokenizer  # noqa: E402

ARTIFACTS = pathlib.Path(__file__).parent.parent / "artifacts" / "bert_tiny"

tokenizer = BertWordPieceTokenizer(str(ARTIFACTS / "vocab.txt"), lowercase=True)
so = ort.SessionOptions()
so.intra_op_num_threads = 1
session = ort.InferenceSession(str(ARTIFACTS / "model.onnx"), sess_options=so, providers=["CPUExecutionProvider"])

cold_start_ms = (time.perf_counter() - t_start) * 1000.0

text = "The quick brown fox jumps over the lazy dog."
enc = tokenizer.encode(text)
ids = np.array([enc.ids], dtype=np.int64)
attention_mask = np.ones_like(ids)
token_type_ids = np.zeros_like(ids)
outputs = session.run(None, {"input_ids": ids, "attention_mask": attention_mask,
                              "token_type_ids": token_type_ids})

if not ONCE:
    print(f"cold_start_ms={cold_start_ms:.2f}  pooled[:5]={outputs[1][0][:5].tolist()}")
