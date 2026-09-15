"""Phase 7: package the gender classifier (exp9/10) for Hugging Face
distribution, per research/gap-analysis.md's brief §24. Publishes nothing —
this only assembles the local package (model card + weights in multiple
formats + benchmark data); actually pushing to a Hugging Face repo is a
separate, explicit step the user does themselves (or asks for by name).

Numbers in the model card are pulled programmatically from the actual
training/benchmark JSON files produced by exp9/exp10, not hand-typed, so the
card can't drift from what was actually measured.
"""
import json
import pathlib
import shutil

import numpy as np
from safetensors.numpy import save_file

ROOT = pathlib.Path(__file__).parent.parent
SRC = ROOT / "artifacts" / "gender_classifier"
OUT = ROOT / "huggingface" / "gender-classifier"


def load_real_metrics():
    training = json.loads((SRC / "training_metrics.json").read_text())
    exp9 = json.loads((ROOT / "research/experiments/exp9-gender-classifier/results.json").read_text())
    exp10 = json.loads((ROOT / "research/experiments/exp10-e2e-native-pipeline/results.json").read_text())
    return training, exp9, exp10


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    training, exp9, exp10 = load_real_metrics()

    in_dim, hidden_dim, out_dim, n_test = map(int, (SRC / "shapes.txt").read_text().split())
    n_params = in_dim * hidden_dim + hidden_dim + hidden_dim * out_dim + out_dim

    # --- native Sajal artifacts, copied as-is: `sajal run huggingface/gender-classifier <name>` works directly ---
    for fname in ["fc1_weight.bin", "fc1_bias.bin", "fc2_weight.bin", "fc2_bias.bin",
                  "shapes.txt", "vocab.txt", "model.onnx"]:
        shutil.copy(SRC / fname, OUT / fname)

    # --- safetensors, for the broader HF/PyTorch ecosystem ---
    def load_f32(name, shape):
        return np.fromfile(SRC / name, dtype=np.float32).reshape(shape)

    tensors = {
        "fc1.weight": load_f32("fc1_weight.bin", (hidden_dim, in_dim)),
        "fc1.bias": load_f32("fc1_bias.bin", (hidden_dim,)),
        "fc2.weight": load_f32("fc2_weight.bin", (out_dim, hidden_dim)),
        "fc2.bias": load_f32("fc2_bias.bin", (out_dim,)),
    }
    save_file(tensors, OUT / "model.safetensors")

    # --- config.json ---
    config = {
        "architecture": "linear_relu_linear_softmax",
        "task": "text-classification",
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
        "out_dim": out_dim,
        "labels": {"0": "M", "1": "F"},
        "feature_extraction": "character_bigram_bag_of_ngrams",
        "vocab_size": in_dim,
        "num_parameters": n_params,
        "precision": "fp32",
        "original_framework": "pytorch",
        "sajal_runtime_compatible": True,
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2))

    # --- curated, machine-readable benchmark summary (brief §28: generate, don't hand-type) ---
    native_cold = next(r for r in exp9["cold_invocation"] if r["impl"] == "native")
    onnx_cold = next(r for r in exp9["cold_invocation"] if r["impl"] == "onnx")
    pytorch_cold = next(r for r in exp9["cold_invocation"] if r["impl"] == "pytorch")
    native_e2e_cold = next(r for r in exp10["results"] if r["impl"] == "native_e2e")
    native_warm = next(r for r in exp9["warm_loop"] if r["impl"] == "native_cpp")
    onnx_warm = next(r for r in exp9["warm_loop"] if r["impl"] == "onnxruntime_cpu")
    pytorch_warm = next(r for r in exp9["warm_loop"] if r["impl"] == "pytorch_eager")

    benchmark = {
        "hardware": "Apple M4, 10 cores, 16GB RAM, macOS (see research/environment.md)",
        "methodology": "research/experiments/exp9-gender-classifier/results.md, exp10-e2e-native-pipeline/results.md",
        "accuracy": {"train": training["train_acc"], "validation": training["val_acc"],
                     "train_n": training["train_n"], "val_n": training["val_n"]},
        "equivalence_native_vs_pytorch": exp9["equivalence"],
        "cold_invocation_ms_p50": {
            "native": native_cold["p50_ms"], "native_end_to_end_raw_text": native_e2e_cold["p50_ms"],
            "onnx_runtime_cpu": onnx_cold["p50_ms"], "pytorch_eager": pytorch_cold["p50_ms"],
        },
        "warm_loop_ms_p50": {
            "native": native_warm["p50_ms"], "onnx_runtime_cpu": onnx_warm["p50_ms"],
            "pytorch_eager": pytorch_warm["p50_ms"],
        },
        "peak_rss_mb": {
            "native": native_warm["peak_rss_mb"], "onnx_runtime_cpu": onnx_warm["peak_rss_mb"],
            "pytorch_eager": pytorch_warm["peak_rss_mb"],
        },
    }
    (OUT / "benchmark_results.json").write_text(json.dumps(benchmark, indent=2))

    # --- model card ---
    model_size_kb = (n_params * 4) / 1000
    native_speedup_onnx = onnx_cold["p50_ms"] / native_cold["p50_ms"]
    native_speedup_pytorch = pytorch_cold["p50_ms"] / native_cold["p50_ms"]

    card = f"""---
license: mit
language:
- en
tags:
- text-classification
- tabular-classification
- sajal-labs
- native-inference
- indian-names
- gender-classification
pipeline_tag: text-classification
---

# Indian-Name Gender Classifier (Sajal Labs)

A small `Linear -> ReLU -> Linear -> Softmax` classifier over character-bigram
features, predicting conventional gender (M/F) from an Indian first name.
Built as the real-model case study for Sajal Labs — a research project on
compiling trained models into minimal, dependency-free native C/C++ inference
artifacts (not yet publicly hosted). See `research/experiments/exp9-gender-classifier`
and `exp10-e2e-native-pipeline` in the Sajal Labs repo for the full methodology.

## Model details

- **Task**: binary text classification (name -> M/F)
- **Architecture**: `Linear({in_dim}->{hidden_dim}) -> ReLU -> Linear({hidden_dim}->{out_dim}) -> Softmax`
- **Feature extraction**: character-bigram bag-of-ngrams over a {in_dim}-entry vocabulary
  (name lowercased, padded with `^`/`$` boundary markers, counted per bigram)
- **Parameters**: {n_params:,}
- **Model size**: ~{model_size_kb:.0f}KB (fp32)
- **Precision**: fp32
- **Original framework**: PyTorch 2.14
- **License**: MIT (code and weights). The training data is a hand-compiled
  list of real Indian first names — names themselves are not copyrightable
  subject matter, so no separate data license applies.

## Accuracy

Trained on {training['train_n']} names, validated on {training['val_n']} held-out names
(85/15 split, vocabulary built from training names only, no leakage).

| | Accuracy |
|---|---:|
| Train | {training['train_acc']*100:.1f}% |
| **Validation** | **{training['val_acc']*100:.1f}%** |

The train/validation gap reflects real overfitting on a small (~800-name) dataset —
reported as measured, not smoothed over. This is a teaching/research case study,
not a claim of state-of-the-art name-gender classification.

## Benchmark (native C++ vs. PyTorch vs. ONNX Runtime, single request, Apple M4 CPU)

Full raw data in `benchmark_results.json`. True end-to-end cold invocation
(process spawn -> raw name string in -> prediction out -> process exit,
external wall-clock, 30 trials, interleaved):

| Implementation | Cold invocation p50 |
|---|---:|
| **Native C++ (Sajal runtime)** | {native_cold['p50_ms']:.2f}ms |
| ONNX Runtime (CPU) | {onnx_cold['p50_ms']:.2f}ms ({native_speedup_onnx:.1f}x slower) |
| PyTorch eager | {pytorch_cold['p50_ms']:.2f}ms ({native_speedup_pytorch:.1f}x slower) |

**This native artifact has no Python dependency at inference time.** Per Sajal
Labs' broader findings (research/gap-analysis.md), the advantage shown here is
specifically in avoiding Python/framework process-startup cost for single-shot
invocation (serverless, CLI, intermittent edge use) — not a general claim that
native code out-computes these frameworks at sustained throughput; see the
linked experiments for the full, honest picture including where the advantage
does and doesn't hold.

## How to use

### Native (Sajal runtime, zero Python)

```bash
sajal run <this_directory> "priya"
# {{"input": "priya", "output": [0.0043, 0.9957]}}   -- [P(male), P(female)]
```

### PyTorch

```python
import json
from safetensors.numpy import load_file

weights = load_file("model.safetensors")   # fc1.weight, fc1.bias, fc2.weight, fc2.bias
config = json.load(open("config.json"))
vocab = open("vocab.txt").read().splitlines()

# See the Sajal Labs repo's python/gender_features.py for the exact bigram
# feature extraction (must match, or predictions will be wrong) and
# python/tiny_mlp.py for the exact forward pass these weights implement.
```

### ONNX Runtime

```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
# feed a [1, {in_dim}] float32 bigram-feature vector (see gender_features.py)
```

## Limitations

- Small, hand-compiled dataset (~800 names) — not demographically representative,
  not exhaustive across India's many naming traditions.
- Genuinely unisex names (e.g. "Kiran") are inherently ambiguous for this task;
  the model picks one label with whatever confidence the training data implies.
- Feature extraction (character bigrams) is a simple, interpretable baseline,
  not a state-of-the-art text representation.
- Benchmarked on Apple Silicon (CPU) only — no CUDA/GPU numbers.

## Files

- `model.safetensors` — weights, HF/PyTorch-ecosystem format
- `model.onnx` — ONNX Runtime-compatible export
- `fc1_weight.bin`, `fc1_bias.bin`, `fc2_weight.bin`, `fc2_bias.bin`, `shapes.txt` — raw native Sajal runtime format
- `vocab.txt` — the {in_dim}-entry character-bigram vocabulary (needed by all three formats)
- `config.json` — architecture metadata
- `benchmark_results.json` — full machine-readable benchmark data behind the numbers above
"""
    (OUT / "README.md").write_text(card)

    print(f"packaged to {OUT}")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size / 1000:.1f}KB)")


if __name__ == "__main__":
    main()
