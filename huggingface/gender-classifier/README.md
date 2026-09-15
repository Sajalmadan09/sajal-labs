---
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
Built as the real-model case study for [Sajal Labs](https://github.com/Sajalmadan09/sajal-labs) —
a research project on compiling trained models into minimal, dependency-free
native C/C++ inference artifacts. See
[exp9-gender-classifier](https://github.com/Sajalmadan09/sajal-labs/tree/main/research/experiments/exp9-gender-classifier)
and [exp10-e2e-native-pipeline](https://github.com/Sajalmadan09/sajal-labs/tree/main/research/experiments/exp10-e2e-native-pipeline)
for the full methodology.

## Model details

- **Task**: binary text classification (name -> M/F)
- **Architecture**: `Linear(343->64) -> ReLU -> Linear(64->2) -> Softmax`
- **Feature extraction**: character-bigram bag-of-ngrams over a 343-entry vocabulary
  (name lowercased, padded with `^`/`$` boundary markers, counted per bigram)
- **Parameters**: 22,146
- **Model size**: ~89KB (fp32)
- **Precision**: fp32
- **Original framework**: PyTorch 2.14
- **License**: MIT (code and weights). The training data is a hand-compiled
  list of real Indian first names — names themselves are not copyrightable
  subject matter, so no separate data license applies.

## Accuracy

Trained on 688 names, validated on 121 held-out names
(85/15 split, vocabulary built from training names only, no leakage).

| | Accuracy |
|---|---:|
| Train | 99.4% |
| **Validation** | **81.8%** |

The train/validation gap reflects real overfitting on a small (~800-name) dataset —
reported as measured, not smoothed over. This is a teaching/research case study,
not a claim of state-of-the-art name-gender classification.

## Benchmark (native C++ vs. PyTorch vs. ONNX Runtime, single request, Apple M4 CPU)

Full raw data in `benchmark_results.json`. True end-to-end cold invocation
(process spawn -> raw name string in -> prediction out -> process exit,
external wall-clock, 30 trials, interleaved):

| Implementation | Cold invocation p50 |
|---|---:|
| **Native C++ (Sajal runtime)** | 3.80ms |
| ONNX Runtime (CPU) | 75.52ms (19.9x slower) |
| PyTorch eager | 584.54ms (153.8x slower) |

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
# {"input": "priya", "output": [0.0043, 0.9957]}   -- [P(male), P(female)]
```

### PyTorch

```python
import json
from safetensors.numpy import load_file

weights = load_file("model.safetensors")   # fc1.weight, fc1.bias, fc2.weight, fc2.bias
config = json.load(open("config.json"))
vocab = open("vocab.txt").read().splitlines()

# See https://github.com/Sajalmadan09/sajal-labs/blob/main/python/gender_features.py
# for the exact bigram feature extraction (must match, or predictions will
# be wrong) and python/tiny_mlp.py for the exact forward pass these weights
# implement.
```

### ONNX Runtime

```python
import onnxruntime as ort
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
# feed a [1, 343] float32 bigram-feature vector (see gender_features.py)
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
- `vocab.txt` — the 343-entry character-bigram vocabulary (needed by all three formats)
- `config.json` — architecture metadata
- `benchmark_results.json` — full machine-readable benchmark data behind the numbers above
