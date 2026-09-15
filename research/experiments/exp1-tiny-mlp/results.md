# Experiment 1 — Tiny MLP: PyTorch vs. Native C++ (Accelerate)

> **Correction (found while building exp7)**: the original `bench_onnx.py` imported `IN_DIM` from `tiny_mlp.py`, which imports `torch` at module scope — so the "ONNX Runtime" cold-start and memory numbers below were silently paying PyTorch's import cost too, on top of ONNX Runtime's own. Fixed by reading `in_dim` from `shapes.txt` directly (no torch dependency), matching the pattern `bench_onnx_transformer.py` already used. **Cold start corrected from ~431-479ms to ~48ms; peak RSS corrected from ~238MB to ~58MB.** Warm per-call latency was unaffected (the bug only inflated one-time import cost, not the measured loop) — those numbers stand. Table below reflects the corrected run; the analysis in §3 has been updated accordingly. This is left visible, not silently edited away, per the principle of showing what failed and why.

**Model**: `softmax(relu(x @ W1^T + b1) @ W2^T + b2)`, dims 64→128→10, random weights (seed 0).
**Hardware**: Apple M4, 10 cores, 16GB RAM, macOS 26.3 (see [environment.md](../../environment.md)).
**Compiler**: Apple clang 17.0.0, `-O3 -std=c++17 -DACCELERATE_NEW_LAPACK -framework Accelerate`.
**Python**: 3.13.2, PyTorch 2.14.0, ONNX Runtime 1.30.0, `torch.set_num_threads(1)` / `intra_op_num_threads=1`.
**Batch size**: 1 (single-request inference — per the brief's RQ6, where native is hypothesized to help most).
**Methodology**: 50 warmup iterations, 500 measured iterations, `p50`/`p90`/`p95`/`p99` via linear-interpolation percentiles (same method in both C++ and Python, so the numbers are comparable). Cold start = process start to model-ready. Peak RSS via `/usr/bin/time -l`. Code: [python/](../../../python/), [native/mlp.cpp](../../../native/mlp.cpp). Raw numbers: [results.json](results.json).

## 1. Numerical equivalence (native C++ vs. PyTorch reference)

| Metric | Value |
|---|---|
| Max absolute error | 4.47e-08 |
| Mean absolute error | 6.17e-09 |
| RMSE | 9.11e-09 |
| Mean cosine similarity | 1.0 |
| Prediction agreement | 100% (20/20 test vectors) |

At fp32 machine epsilon (~1.19e-7), this is as close to "identical" as fp32 arithmetic allows — see [equivalence_result.json](../../../artifacts/equivalence_result.json). **Observation, not yet verified**: this is tighter agreement than IEEE-754 non-associativity alone would predict for two independently-written matmul implementations (research/papers.md §8) — plausible explanation is that PyTorch's own CPU backend also routes through Accelerate/vecLib on Apple Silicon, so both sides may be calling the same underlying BLAS kernel. Worth confirming later (e.g. via `torch.config.show()` or tracing), not confirmed in this experiment.

## 2. Benchmark results

| Implementation | Cold start (ms) | Warm p50 (ms) | Warm p99 (ms) | Peak RSS (MB) | On-disk footprint (MB) |
|---|---:|---:|---:|---:|---:|
| **Native C++ (Accelerate)** | 0.77 | 0.00079 | 0.00083 | 5.80 | 0.04 (binary) |
| PyTorch eager | 711.55 | 0.00921 | 0.00963 | 217.28 | 535.78 (site-packages) |
| ONNX Runtime (CPU EP) | 47.98 | 0.00546 | 0.00655 | 58.11 | 78.28 (site-packages) |

## 3. Interpretation — tied back to the brief's research questions

- **RQ2 (fidelity)**: fully preserved at this tolerance — no accuracy/behavior tradeoff observed for this model.
- **RQ3/RQ5 (latency, when native wins)**: two very different effects are visible here, and conflating them would overclaim:
  - **Cold start is where the gap is enormous, and PyTorch and ONNX Runtime are NOT equally slow here** (a distinction the pre-correction numbers had masked): native 0.77ms, ONNX Runtime 47.98ms (~62x), PyTorch eager 711.55ms (~924x, ~15x slower than ONNX Runtime itself). This is **not primarily an ML-inference effect** — it's the cost of starting a Python process and importing its C-extension libraries before any inference happens (torch alone: 535MB on disk, and evidently a much heavier import than onnxruntime's 78MB). This cost disappears in a long-running server process that imports once and handles many requests — so it matters enormously for **serverless/CLI/edge/one-shot invocation** patterns (RQ6) and barely at all for a warm, persistent server.
  - **Warm per-call latency is where genuine per-inference overhead shows up**, and native is still faster but by a much smaller, more defensible margin: ~6.9x vs. ONNX Runtime CPU (0.00079ms vs 0.00546ms p50), ~11.7x vs. PyTorch eager. For a model this tiny (single-digit microseconds of actual compute), Python/framework dispatch overhead (object allocation, `torch.no_grad()` context, session.run() marshaling) very plausibly dominates over the matmul itself — this experiment doesn't yet isolate how much of this is "dispatch overhead" vs. "our C++ path skips work ONNX Runtime's generic graph executor still does." (exp2-6 later found this warm-latency edge does not survive to transformer-shaped models — see [gap-analysis.md](../../gap-analysis.md)'s revised recommendation.)
- **RQ4 (memory/dependency)**: peak RSS is ~10x smaller than ONNX Runtime (5.80MB vs 58.11MB) and ~37x smaller than PyTorch (vs 217.28MB); on-disk dependency footprint is 78-536MB (Python) vs. 0.04MB (one static-ish binary). This is a real, measured, reproducible number — one of the more defensible claims from this experiment, and one that held up across every later experiment too (see exp2-7).
- **Caveats that limit generalization**: one model, one (tiny) size, one machine (Apple Silicon, no CUDA comparison possible per environment.md), one run. No multi-threading, no batching beyond size 1, no larger model tested. This experiment establishes the methodology and gets an honest first data point — it does not establish that these ratios hold for larger/different models, and shouldn't be cited as if it does.

## 4. What this experiment does and doesn't prove

**Establishes**: the equivalence-measurement harness works and produces a real, tight, explainable tolerance; the benchmark harness produces fair, reproducible, machine-documented numbers; for this trivial model, native C++ wins decisively on cold start and dependency footprint, and meaningfully (not overwhelmingly) on warm per-call latency.

**Does not establish**: that this holds for larger models, batched inference, multi-threaded serving, or non-Apple-Silicon hardware; that the warm-latency gap is due to "C++ vs. Python" rather than "hand-written minimal dispatch vs. general-purpose graph executor" (ONNX Runtime is also C++ under the hood — the gap there is architectural, not language-based, and should be described that way).

## 5. Suggested next experiment

Repeat this exact harness on a model 10-100x larger (still non-LLM — e.g. a small CNN or a 2-3 layer transformer block) to see whether the warm-latency ratio holds, shrinks, or grows as actual compute time starts to dominate over dispatch overhead — this directly tests whether the "native wins most for small/single-request inference" hypothesis (RQ6) is really about model size or was an artifact of picking an unrealistically tiny first model.
