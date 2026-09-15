"""Phase 7 (BERT edition): package the native C++ port of prajjwal1/bert-tiny
for Hugging Face — see export_hf_package.py's docstring for the general
approach (publishes nothing itself; numbers pulled from real JSON, not
hand-typed). The critical difference from the gender-classifier package:
THIS IS NOT OUR MODEL. prajjwal1/bert-tiny is a third-party MIT-licensed
pretrained checkpoint (Turc et al. 2019, Bhargava et al. 2021) — our
contribution is the native C++ port (encoder + WordPiece tokenizer) and
the equivalence/benchmark validation, not the weights themselves. The card
generated here credits the original authors prominently and first, states
the required citations, and is explicit throughout about which parts are
ours vs. theirs.
"""
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).parent.parent
SRC = ROOT / "artifacts" / "bert_tiny"
OUT = ROOT / "huggingface" / "bert-tiny-native"

BASE_MODEL = "prajjwal1/bert-tiny"


def load_real_metrics():
    equiv = json.loads((SRC / "equivalence_result.json").read_text())
    exp11_cold = json.loads(
        (ROOT / "research/experiments/exp11-real-pretrained-transformer/cold_invocation_results.json").read_text()
    )
    exp11_warm = json.loads(
        (ROOT / "research/experiments/exp11-real-pretrained-transformer/warm_loop_results.json").read_text()
    )
    exp14_cold = json.loads(
        (ROOT / "research/experiments/exp14-wordpiece-native/cold_invocation_results.json").read_text()
    )
    return equiv, exp11_cold, exp11_warm, exp14_cold


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    equiv, exp11_cold, exp11_warm, exp14_cold = load_real_metrics()

    num_layers, hidden, num_heads, intermediate, vocab_size, max_pos, eps = (
        SRC / "config.txt"
    ).read_text().split()
    n_params = 4_385_920  # from exp11's export log — the real prajjwal1/bert-tiny parameter count

    # --- native Sajal artifacts (encoder weights + WordPiece vocab/config), copied as-is ---
    native_files = ["config.txt", "vocab.txt", "tokenizer_config.txt",
                     "word_embeddings.bin", "position_embeddings.bin", "token_type_embeddings.bin",
                     "emb_ln_weight.bin", "emb_ln_bias.bin", "pooler_weight.bin", "pooler_bias.bin",
                     "model.onnx", "model.onnx.data"]
    for i in range(int(num_layers)):
        for suffix in ["q_weight", "q_bias", "k_weight", "k_bias", "v_weight", "v_bias",
                        "attn_out_weight", "attn_out_bias", "attn_ln_weight", "attn_ln_bias",
                        "ff1_weight", "ff1_bias", "ff2_weight", "ff2_bias", "ff_ln_weight", "ff_ln_bias"]:
            native_files.append(f"layer{i}_{suffix}.bin")
    for fname in native_files:
        shutil.copy(SRC / fname, OUT / fname)

    # --- safetensors, for the broader HF/PyTorch ecosystem (via transformers' own save) ---
    import torch
    from transformers import BertModel
    from safetensors.torch import save_file
    model = BertModel.from_pretrained(BASE_MODEL).eval()
    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, OUT / "model.safetensors")

    # --- config.json ---
    config = {
        "base_model": BASE_MODEL, "base_model_license": "MIT",
        "architecture": "bert", "num_hidden_layers": int(num_layers), "hidden_size": int(hidden),
        "num_attention_heads": int(num_heads), "intermediate_size": int(intermediate),
        "vocab_size": int(vocab_size), "max_position_embeddings": int(max_pos),
        "layer_norm_eps": float(eps), "num_parameters": n_params, "precision": "fp32",
        "original_framework": "pytorch", "sajal_runtime_compatible": True,
        "sajal_native_port": {
            "encoder": "native/bert_model.hpp", "tokenizer": "native/wordpiece.hpp (real WordPiece, ASCII-scoped)",
        },
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2))

    # --- curated, machine-readable benchmark + equivalence summary ---
    native_cold = next(r for r in exp11_cold["results"] if r["impl"] == "native")
    onnx_cold = next(r for r in exp11_cold["results"] if r["impl"] == "onnx")
    pytorch_cold = next(r for r in exp11_cold["results"] if r["impl"] == "pytorch_transformers")
    native_warm = next(r for r in exp11_warm["results"] if r["impl"] == "native_cpp")
    onnx_warm = next(r for r in exp11_warm["results"] if r["impl"] == "onnxruntime_cpu")
    pytorch_warm = next(r for r in exp11_warm["results"] if r["impl"] == "pytorch_eager")
    native_e2e = next(r for r in exp14_cold["results"] if r["impl"] == "native_e2e")
    onnx_lean_e2e = next(r for r in exp14_cold["results"] if r["impl"] == "onnx_e2e_lean")
    onnx_tf_e2e = next(r for r in exp14_cold["results"] if r["impl"] == "onnx_e2e_transformers")
    pytorch_e2e = next(r for r in exp14_cold["results"] if r["impl"] == "pytorch_e2e")

    REPO = "https://github.com/Sajalmadan09/sajal-labs/tree/main"
    benchmark = {
        "hardware": f"Apple M4, 10 cores, 16GB RAM, macOS (see {REPO}/research/environment.md)",
        "methodology": (f"{REPO}/research/experiments/exp11-real-pretrained-transformer/results.md, "
                         f"{REPO}/research/experiments/exp12-warm-latency-width-depth/results.md, "
                         f"{REPO}/research/experiments/exp13-width-threshold/results.md, "
                         f"{REPO}/research/experiments/exp14-wordpiece-native/results.md"),
        "equivalence_native_vs_pytorch": {
            "note": "native encoder + tokenizer vs. HF BertModel + BertTokenizerFast, 10 real sentences (lengths 4-25 tokens)",
            "hidden_state_max_abs_error": equiv["hidden_max_abs_error_overall"],
            "pooled_output_max_abs_error": equiv["pooled_max_abs_error_overall"],
            "mean_cosine_similarity_pooled": equiv["mean_pooled_cos_sim"],
        },
        "tokenizer_equivalence": "17/17 test sentences (incl. contractions, hyphens, OOV words, emails) produced byte-identical token IDs to HF's real tokenizer — see exp14",
        "cold_invocation_ms_p50_precomputed_tokens": {
            "native": native_cold["p50_ms"], "onnx_runtime_cpu": onnx_cold["p50_ms"],
            "pytorch_plus_transformers": pytorch_cold["p50_ms"],
        },
        "cold_invocation_ms_p50_raw_text_full_pipeline": {
            "native": native_e2e["p50_ms"], "onnx_runtime_cpu_plus_lean_tokenizer": onnx_lean_e2e["p50_ms"],
            "onnx_runtime_cpu_plus_transformers_tokenizer": onnx_tf_e2e["p50_ms"],
            "pytorch_plus_transformers": pytorch_e2e["p50_ms"],
        },
        "warm_loop_ms_p50": {
            "native": native_warm["p50_ms"], "onnx_runtime_cpu": onnx_warm["p50_ms"],
            "pytorch_eager": pytorch_warm["p50_ms"],
        },
        "warm_loop_note": ("Unlike cold invocation, native's warm-loop edge is NOT unconditional — "
                            "it depends on model width; see exp12/exp13. At this model's hidden_size=128 "
                            "it's below the measured ~250 crossover, so native keeps a real edge."),
    }
    (OUT / "benchmark_results.json").write_text(json.dumps(benchmark, indent=2))

    # --- model card ---
    native_speedup_onnx_e2e = onnx_lean_e2e["p50_ms"] / native_e2e["p50_ms"]
    native_speedup_onnx_tf_e2e = onnx_tf_e2e["p50_ms"] / native_e2e["p50_ms"]
    native_speedup_pytorch_e2e = pytorch_e2e["p50_ms"] / native_e2e["p50_ms"]

    card = f"""---
license: mit
language:
- en
tags:
- bert
- native-inference
- sajal-labs
- transformer
- feature-extraction
base_model: {BASE_MODEL}
base_model_relation: finetune
pipeline_tag: feature-extraction
---

# bert-tiny — Native C++ Port (Sajal Labs)

**This is not a new model.** The weights, architecture, and pretraining are
entirely [`{BASE_MODEL}`](https://huggingface.co/{BASE_MODEL}) by Prajjwal
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

- **Architecture**: BERT encoder, {num_layers} layers, hidden={hidden}, heads={num_heads}, intermediate={intermediate}
- **Vocabulary**: {vocab_size} WordPiece tokens (bert-base-uncased vocab)
- **Parameters**: {n_params:,}
- **Precision**: fp32
- **Base model license**: MIT ({BASE_MODEL})
- **Port license**: MIT (Sajal Labs' C++ code)

## What was verified (Sajal Labs' contribution)

**Tokenizer**: 17/17 real test sentences — including contractions ("don't"),
hyphenation ("COVID-19"), an out-of-vocabulary word forcing an 11-piece
subword split, and an email address — produced **byte-identical token IDs**
to the original `BertTokenizerFast`. This is an exact-match bar, not a
tolerance: tokenization is deterministic.

**Encoder**: max absolute error {equiv['hidden_max_abs_error_overall']:.2e} (hidden states),
{equiv['pooled_max_abs_error_overall']:.2e} (pooled `[CLS]` output), cosine similarity
~1.0, across 10 real sentences of varying length (4-25 tokens) — fp32-scale
agreement, consistent with floating-point non-associativity between two
independent implementations (not a bug; see the repo's `research/papers.md`).

## Benchmark (single request, Apple M4 CPU — full data in `benchmark_results.json`)

True end-to-end cold invocation (process spawn → raw text in → prediction out
→ process exit, external wall-clock):

| Implementation | Cold invocation p50 |
|---|---:|
| **Native C++ (Sajal runtime)** | {native_e2e['p50_ms']:.2f}ms |
| ONNX Runtime + lean tokenizer | {onnx_lean_e2e['p50_ms']:.2f}ms ({native_speedup_onnx_e2e:.1f}x slower) |
| ONNX Runtime + 🤗 transformers tokenizer | {onnx_tf_e2e['p50_ms']:.2f}ms ({native_speedup_onnx_tf_e2e:.1f}x slower) |
| PyTorch + 🤗 transformers | {pytorch_e2e['p50_ms']:.2f}ms ({native_speedup_pytorch_e2e:.1f}x slower) |

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
model = BertModel.from_pretrained("{BASE_MODEL}")
tokenizer = BertTokenizerFast.from_pretrained("{BASE_MODEL}")
```

### ONNX Runtime

```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
# feed input_ids/attention_mask/token_type_ids from any WordPiece tokenizer
```

## Citations (required if you use this model)

```bibtex
@article{{turc2019distillation,
  title={{Well-Read Students Learn Better: On the Importance of Pre-training Compact Models}},
  author={{Turc, Iulia and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina}},
  journal={{arXiv preprint arXiv:1908.08962v2}},
  year={{2019}}
}}

@misc{{bhargava2021generalization,
  title={{Generalization in NLI: Ways (Not) To Go Beyond Simple Heuristics}},
  author={{Bhargava, Prajjwal and Drozd, Aleksandr and Rogers, Anna}},
  year={{2021}},
  eprint={{2110.01518}},
  archivePrefix={{arXiv}},
  primaryClass={{cs.CL}}
}}
```

## Files

- `model.safetensors` — the original weights, HF/PyTorch-ecosystem format
- `model.onnx` + `model.onnx.data` — ONNX Runtime-compatible export (weights externalized to the `.data` file; both are required together)
- `word_embeddings.bin`, `position_embeddings.bin`, `token_type_embeddings.bin`, `emb_ln_*.bin`, `layer{{i}}_*.bin`, `pooler_*.bin` — raw native Sajal runtime format
- `vocab.txt`, `tokenizer_config.txt` — WordPiece vocabulary + config for the native tokenizer port
- `config.json` — architecture metadata
- `benchmark_results.json` — full machine-readable benchmark/equivalence data behind the numbers above
"""
    (OUT / "README.md").write_text(card)

    print(f"packaged to {OUT}")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size / 1000:.1f}KB)")


if __name__ == "__main__":
    main()
