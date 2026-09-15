---
license: mit
language:
- en
tags:
- bert
- native-inference
- sajal-labs
- transformer
- feature-extraction
base_model: prajjwal1/bert-tiny
base_model_relation: finetune
pipeline_tag: feature-extraction
---

# bert-tiny — Native C++ Port (Sajal Labs)

**This is not a new model.** The weights, architecture, and pretraining are
entirely [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny) by Prajjwal
Bhargava (MIT license) — a compact pretrained BERT encoder introduced in
Turc et al. 2019 ("Well-Read Students Learn Better") and ported to
Hugging Face for Bhargava et al. 2021 ("Generalization in NLI"). **Please
cite both papers if you use this model** (citations below).

**What Sajal Labs added**: a from-scratch native C++ port of the encoder
and the WordPiece tokenizer — no PyTorch, no `transformers`, no Python at
inference time — with rigorous equivalence and benchmark validation against
the original. See [Sajal Labs](https://github.com/Sajalmadan09/sajal-labs),
experiments
[exp11](https://github.com/Sajalmadan09/sajal-labs/tree/main/research/experiments/exp11-real-pretrained-transformer)
through
[exp14](https://github.com/Sajalmadan09/sajal-labs/tree/main/research/experiments/exp14-wordpiece-native),
for full methodology.

## Model details (unchanged from the original)

- **Architecture**: BERT encoder, 2 layers, hidden=128, heads=2, intermediate=512
- **Vocabulary**: 30522 WordPiece tokens (bert-base-uncased vocab)
- **Parameters**: 4,385,920
- **Precision**: fp32
- **Base model license**: MIT (prajjwal1/bert-tiny)
- **Port license**: MIT (Sajal Labs' C++ code)

## What was verified (Sajal Labs' contribution)

**Tokenizer**: 17/17 real test sentences — including contractions ("don't"),
hyphenation ("COVID-19"), an out-of-vocabulary word forcing an 11-piece
subword split, and an email address — produced **byte-identical token IDs**
to the original `BertTokenizerFast`. This is an exact-match bar, not a
tolerance: tokenization is deterministic.

**Encoder**: max absolute error 9.54e-06 (hidden states),
2.19e-06 (pooled `[CLS]` output), cosine similarity
~1.0, across 10 real sentences of varying length (4-25 tokens) — fp32-scale
agreement, consistent with floating-point non-associativity between two
independent implementations (not a bug; see the repo's `research/papers.md`).

## Benchmark (single request, Apple M4 CPU — full data in `benchmark_results.json`)

True end-to-end cold invocation (process spawn → raw text in → prediction out
→ process exit, external wall-clock):

| Implementation | Cold invocation p50 |
|---|---:|
| **Native C++ (Sajal runtime)** | 11.37ms |
| ONNX Runtime + lean tokenizer | 95.72ms (8.4x slower) |
| ONNX Runtime + 🤗 transformers tokenizer | 2495.61ms (219.6x slower) |
| PyTorch + 🤗 transformers | 4973.91ms (437.6x slower) |

Worth knowing before you read too much into "ONNX Runtime": its own cold-start
number depends heavily on which tokenizer library it's paired with — using
`transformers` for convenience costs ~25x more than using the lean, standalone
`tokenizers` library for the exact same token IDs. Native sidesteps that
whole dependency-choice question by construction. Full discussion in
[exp14](https://github.com/Sajalmadan09/sajal-labs/tree/main/research/experiments/exp14-wordpiece-native).

**Honest scope note on the warm-loop numbers**: native's advantage is
*not* unconditional the way cold-invocation is — Sajal Labs found it
depends on model width (`hidden_size`), with a measured crossover around
`hidden≈250` on this hardware
([exp12](https://github.com/Sajalmadan09/sajal-labs/tree/main/research/experiments/exp12-warm-latency-width-depth)/[exp13](https://github.com/Sajalmadan09/sajal-labs/tree/main/research/experiments/exp13-width-threshold)).
This model's `hidden=128`
sits comfortably below that, so native keeps a real warm-loop edge too —
but that's a property of this model's size, not a general claim.

## How to use

### Native (Sajal runtime, zero Python)

```bash
sajal run <this_directory> "The quick brown fox jumps over the lazy dog."
```

### PyTorch / transformers (the original)

```python
from transformers import BertModel, BertTokenizerFast
model = BertModel.from_pretrained("prajjwal1/bert-tiny")
tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")
```

### ONNX Runtime

```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
# feed input_ids/attention_mask/token_type_ids from any WordPiece tokenizer
```

## Citations (required if you use this model)

```bibtex
@article{turc2019distillation,
  title={Well-Read Students Learn Better: On the Importance of Pre-training Compact Models},
  author={Turc, Iulia and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1908.08962v2},
  year={2019}
}

@misc{bhargava2021generalization,
  title={Generalization in NLI: Ways (Not) To Go Beyond Simple Heuristics},
  author={Bhargava, Prajjwal and Drozd, Aleksandr and Rogers, Anna},
  year={2021},
  eprint={2110.01518},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

## Files

- `model.safetensors` — the original weights, HF/PyTorch-ecosystem format
- `model.onnx` + `model.onnx.data` — ONNX Runtime-compatible export (weights externalized to the `.data` file; both are required together)
- `word_embeddings.bin`, `position_embeddings.bin`, `token_type_embeddings.bin`, `emb_ln_*.bin`, `layer{i}_*.bin`, `pooler_*.bin` — raw native Sajal runtime format
- `vocab.txt`, `tokenizer_config.txt` — WordPiece vocabulary + config for the native tokenizer port
- `config.json` — architecture metadata
- `benchmark_results.json` — full machine-readable benchmark/equivalence data behind the numbers above
