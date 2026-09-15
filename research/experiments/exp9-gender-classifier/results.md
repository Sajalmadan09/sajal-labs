# Experiment 9 — Real Model Case Study: Indian-Name Gender Classifier

**Why this experiment**: exp1-8 all used hand-built synthetic architectures with random or lightly-trained weights. Per the project brief's Phase 3/Model 4, the equivalence + benchmark methodology needs to be validated on a real trained model doing a real task, not just teaching architectures. This is that: a genuine classifier, trained on genuine data, taken through the exact same equivalence and cold-invocation pipeline as exp1-8.

## The data — real, not fabricated

No existing Indian-name-gender project was found locally or on GitHub (searched both before building this). Built a real dataset instead: [gender_data.py](../../../python/gender_data.py) hand-compiles **809 genuine Indian first names** (450 male, 359 female, 2 dropped as legitimately ambiguous) spanning multiple linguistic/religious naming traditions — North Indian Hindu/Sanskrit-origin, South Indian Tamil/Telugu/Kannada/Malayalam, Muslim, Sikh/Punjabi, Bengali, Gujarati, Marathi — drawn from general knowledge of names in wide use, not scraped or invented. This is a small, honest dataset, not a claim of demographic completeness or statistical representativeness — every name in it is real, but the list isn't exhaustive and wasn't sampled to match population proportions.

## The model — reused architecture, zero new C++ code

[gender_features.py](../../../python/gender_features.py) extracts character-bigram bag-of-ngrams features (e.g. "ravi" → `^r, ra, av, vi, i$`), a classic, well-understood text-classification technique (same family as fastText). This was chosen specifically so the resulting classifier is **architecturally identical to exp1's TinyMLP** (Linear → ReLU → Linear → Softmax) — meaning [native/mlp.cpp](../../../native/mlp.cpp) needed **zero changes** to run this real model; it already reads `in_dim`/`hidden_dim`/`out_dim` from `shapes.txt` at runtime. `tiny_mlp.py`'s `TinyMLP` was parameterized (constructor args instead of module constants, matching the pattern already used for `TinyTransformerBlock`) so [gender_train.py](../../../python/gender_train.py) could reuse it directly with `in_dim=343` (vocab size), `hidden_dim=64`, `out_dim=2` (M/F).

## Training result — honest, including the overfitting

688 training / 121 validation names (85/15 split, vocab built from train only, no leakage), 200 epochs, Adam:

| | Train | Validation |
|---|---:|---:|
| Accuracy | 99.4% | **81.8%** |

The train/val gap is real overfitting on a small dataset (343 features, 688 examples) and is reported as-is, not smoothed over. 81.8% validation accuracy is a genuinely reasonable result for character-pattern-based Indian name gender classification — well above the 50% random baseline, consistent with this being a real, non-trivial task where many names carry genuine, learnable phonetic gender signal (e.g. names ending in "a" skewing female) but others are legitimately ambiguous or region-dependent. A few concrete validation examples, not just the aggregate number:

| Name | True | Predicted | Confidence | |
|---|---|---|---:|---|
| chandran | M | M | 1.000 | correct |
| jayalakshmi | F | F | 0.998 | correct |
| mrunal | F | M | 0.983 | **wrong, confidently** |
| kiran | F | M | 0.865 | wrong — Kiran is a genuinely unisex name in India |
| safia | F | M | 0.606 | wrong, but low confidence — appropriately uncertain |

Full metrics: [artifacts/gender_classifier/training_metrics.json](../../../artifacts/gender_classifier/training_metrics.json) (gitignored, regenerable via `python/gender_train.py`).

## Numerical equivalence (native C++ vs. PyTorch reference)

| Metric | Value |
|---|---|
| Max absolute error | 1.19e-07 |
| Mean absolute error | 5.11e-09 |
| RMSE | 1.52e-08 |
| Prediction agreement | **100%** (121/121 validation names) |

fp32-machine-epsilon-level agreement, consistent with exp1's original MLP equivalence result — the C++ port reproduces this real model's behavior exactly as faithfully as it did the synthetic one.

## Benchmark results

Same methodology as exp7/exp8 for cold invocation (external wall-clock, interleaved trials, 30 trials/config) plus exp1-6's warm-loop methodology for completeness. Full data: [results.json](results.json).

| | Cold invocation p50 | Warm loop p50 | Peak RSS |
|---|---:|---:|---:|
| **Native C++** | 3.80ms | 0.00067ms | 5.90MB |
| ONNX Runtime CPU | 75.52ms (**19.9x**) | 0.00479ms (**7.2x**) | 58.77MB |
| PyTorch eager | 584.54ms (**153.8x**) | 0.00896ms (**13.4x**) | 217.58MB |

## Interpretation

**Cold-invocation numbers land right in the range exp8's sweep predicted** (15.7-19.7x vs. ONNX Runtime, 112-149x vs. PyTorch across `d_model` 16-256) — this classifier's 343-dim input sits near that range's low end, and 19.9x/153.8x is consistent with it. The pivot's central claim transfers cleanly from synthetic sweep models to a real trained model on a real task.

**The warm-loop numbers are the more interesting confirmation.** This classifier uses exp1's simple 2-matmul MLP architecture, not exp2-6's transformer — and its warm-latency ratios (7.2x vs. ONNX Runtime, 13.4x vs. PyTorch) land close to exp1's original 6.2x/12.6x, not exp2-6's ~1x collapse. This is exactly what the accumulated findings predict: the warm-latency advantage was never really about "native vs. Python" in general, it was about *operator/call count* (exp5's finding) — a 2-matmul model stays in the regime where native wins on both axes; a many-small-matmul-call model (the transformer's per-head attention) only wins on the cold-invocation axis. **This real model is additional evidence for exp5's mechanism, not just a repeat of exp7/exp8's headline number.**

## What this experiment establishes

- The full pipeline (real data → real training → equivalence check → cold-invocation benchmark) works end to end on a model nobody hand-tuned for the demo.
- exp7/exp8's cold-invocation claim (15-155x, architecture- and size-independent) holds for a real task, not just synthetic sweeps.
- exp5's "call count, not language" explanation for when warm latency matters gets independent support from a real model landing exactly where that theory predicts.
- Deployment realism, not just weight-porting: this is the first experiment where the *preprocessing* (character-bigram feature extraction) is as real a part of the deployment story as the model itself — a genuine "trained model to native artifact" pipeline needs both ported correctly, not just the learned weights.

## Caveats

Feature extraction (name → bigram vector) is still done in Python for both the equivalence check and the benchmarks — the C++ side receives precomputed `test_inputs.bin`, not raw name strings, same methodology as exp1-8. This means the benchmark doesn't yet measure a true "raw string in, prediction out" native pipeline; the C++ program never has to tokenize/featurize anything, it just runs the matmuls. 81.8% validation accuracy is honest but not competitive with production name-gender classifiers built on much larger datasets — this experiment validates the *systems* methodology (equivalence, benchmarking, packaging), not a claim about state-of-the-art name classification.

## Next experiment

Port the bigram feature extraction itself into C++ (a `std::string` → fixed-size float vector function, reusing the same vocabulary file) so the native binary takes a raw name string as input and produces a prediction with no Python involved anywhere in the pipeline — the first fully end-to-end native deployment in the project, and a direct test of whether preprocessing cost (currently invisible, since it's done once in Python before either benchmark runs) changes the cold-invocation picture once it's part of what gets measured.
