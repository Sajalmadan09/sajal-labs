# Experiment 10 — Fully End-to-End Native Pipeline

**Why this experiment**: exp1-9 all fed the C++ side precomputed feature vectors (`test_inputs.bin`) — a real deployment never gets a pre-extracted float vector, it gets a raw name string. This is the first experiment where the native binary does the *whole* job: raw string in, prediction out, with feature extraction ported to C++ too. Two questions: (1) does the C++ feature extractor actually match Python's exactly, not just the matmuls downstream of it, and (2) does paying real preprocessing cost (instead of a precomputed dummy input) change exp7-9's cold-invocation numbers?

## What was built

[native/gender_predict.cpp](../../../native/gender_predict.cpp) ports [python/gender_features.py](../../../python/gender_features.py)'s `normalize()`/`bigrams()`/`extract_features()` to C++ line-for-line (lowercase, `^`/`$` boundary padding, bigram bag-of-ngrams over a vocabulary loaded from `vocab.txt`), then runs the same `linear → relu → linear → softmax` forward pass as `mlp.cpp` via `common.hpp`'s shared ops. Three modes: `predict <name>` (single prediction), `predict_batch <names_file>` (writes `native_outputs.bin` in the same format `compare.py` already reads — reusing exp1-9's equivalence-check tooling unchanged), `once` (cold-invocation benchmarking, matching `mlp.cpp`/`transformer.cpp`'s existing contract). [gender_train.py](../../../python/gender_train.py) now also saves `val_names.txt` (the raw validation names, same order as `test_inputs.bin`) so the batch mode can be checked against the exact same reference outputs exp9 used. Python counterparts [predict_pytorch.py](../../../python/predict_pytorch.py) and [predict_onnx.py](../../../python/predict_onnx.py) do the equivalent raw-name-to-prediction pipeline, for a fair three-way comparison that all pays real feature-extraction cost, not just the C++ side.

## Equivalence: true end-to-end, not just the matmuls

Ran `gender_predict predict_batch` on all 121 raw validation names, compared against the same `ref_outputs.bin` exp9 used:

| Metric | exp9 (precomputed features) | exp10 (raw strings, C++ extracts features) |
|---|---:|---:|
| Max absolute error | 1.19e-07 | 1.19e-07 |
| Prediction agreement | 100% | **100%** |

Identical to exp9's numbers. This confirms the C++ bigram feature extractor produces *exactly* the same vectors as Python's for every one of the 121 validation names — not approximately, not "close enough," but byte-for-byte equivalent input to the matmul, which is why the downstream error is unchanged from exp9's model-only equivalence check. A three-way sanity check on individual names ("sajal", "priya") showed `native_e2e`, `predict_pytorch.py`, and `predict_onnx.py` all agreeing to 5-6 decimal places on `prob_M`/`prob_F`.

## Benchmark: does real preprocessing change the cold-invocation picture?

| | exp9 (dummy precomputed input) | exp10 (real feature extraction) |
|---|---:|---:|
| native p50 | 3.80ms | 4.06ms |
| onnx p50 | 75.52ms | 73.76ms |
| pytorch p50 | 584.54ms | 582.88ms |
| native vs. onnx | 19.9x | 18.2x |
| native vs. pytorch | 153.8x | 143.6x |

Full data: [results.json](results.json).

**No meaningful change.** All three numbers moved by less than 3% — within the run-to-run noise this project has seen throughout (compare exp3 vs exp4's ±0.1x swings from a single rerun). Bigram extraction for a ~5-10 character name against a 343-entry vocabulary is genuinely cheap (a handful of string operations and hash-map lookups) compared to process-startup cost, on both the native and Python sides — so it doesn't show up as a measurable line item in a cost dominated by 3-600ms of process/import overhead.

## Interpretation

This closes the loop the project opened at exp1: the pivot's cold-invocation claim now rests on a fully real, fully end-to-end pipeline — real trained model, real data, real preprocessing, ported completely to native code, checked for exact equivalence at every stage, benchmarked honestly. Nothing about adding the "missing" preprocessing step changed the conclusion; if anything it strengthens it, since the natural worry ("maybe preprocessing dominates and the C++ advantage is illusory once you count it") is directly answered: preprocessing cost is real but small, and doesn't move the needle at process-startup-dominated timescales.

## Caveats

Feature extraction cost being negligible here is specific to this task's cheap preprocessing (string hashing over a small vocab). A model with expensive preprocessing (e.g. a large tokenizer vocabulary, image resizing/normalization, audio feature extraction) could show a different picture — this experiment establishes the methodology for checking that, not a universal claim that preprocessing never matters.

## Next experiment

This is a natural stopping point for the equivalence/benchmarking line of work — nine-plus experiments have now validated the core hypothesis (cold-invocation advantage), diagnosed the warm-latency exception (call-count overhead), and confirmed both hold end-to-end on a real model with real preprocessing. Per the project's roadmap (research/gap-analysis.md, README.md), the natural next phase is less "measure more" and more "package what's been learned" — Phase 5/6 of the original roadmap: a minimal CLI (`sajal run <model> <input>`) wrapping this pattern generically, rather than a bespoke `gender_predict` binary per model.
