"""exp14's Python end-to-end counterpart to native/bert's once_e2e mode —
real tokenizer + real model, both paid for every invocation, for a fair
cold-invocation comparison against the fully-native pipeline."""
import sys
import time

ONCE = "--once" in sys.argv

t_start = time.perf_counter()

import torch  # noqa: E402
from transformers import BertModel, BertTokenizerFast  # noqa: E402

model = BertModel.from_pretrained("prajjwal1/bert-tiny").eval()
tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")

cold_start_ms = (time.perf_counter() - t_start) * 1000.0

text = "The quick brown fox jumps over the lazy dog."
encoded = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    out = model(**encoded)

if not ONCE:
    print(f"cold_start_ms={cold_start_ms:.2f}  pooled[:5]={out.pooler_output[0][:5].tolist()}")
