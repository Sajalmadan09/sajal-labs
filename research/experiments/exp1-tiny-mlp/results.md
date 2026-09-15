# Experiment 1 — Tiny MLP: PyTorch vs. Native C++ (Accelerate)

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
| **Native C++ (Accelerate)** | 0.11 | 0.00071 | 0.00142 | 5.75 | 0.04 (binary) |
| PyTorch eager | 395.92 | 0.00896 | 0.00958 | 216.71 | 535.78 (site-packages) |
| ONNX Runtime (CPU EP) | 430.93 | 0.00437 | 0.00504 | 237.93 | 78.28 (site-packages) |

## 3. Interpretation — tied back to the brief's research questions

- **RQ2 (fidelity)**: fully preserved at this tolerance — no accuracy/behavior tradeoff observed for this model.
- **RQ3/RQ5 (latency, when native wins)**: two very different effects are visible here, and conflating them would overclaim:
  - **Cold start is where the gap is enormous** (~3,600–4,000x): native is 0.11ms, PyTorch/ONNX Runtime are 400-430ms. This is **not primarily an ML-inference effect** — it's the cost of starting a Python process and importing large C-extension libraries (torch: 535MB on disk; onnxruntime: 78MB) before any inference happens. Any Python program with heavy imports pays this, regardless of what it's importing. This cost disappears in a long-running server process that imports once and handles many requests — so it matters enormously for **serverless/CLI/edge/one-shot invocation** patterns (RQ6) and barely at all for a warm, persistent server.
  - **Warm per-call latency is where genuine per-inference overhead shows up**, and native is still faster but by a much smaller, more defensible margin: ~6x vs. ONNX Runtime CPU (0.00071ms vs 0.00437ms p50), ~13x vs. PyTorch eager. For a model this tiny (single-digit microseconds of actual compute), Python/framework dispatch overhead (object allocation, `torch.no_grad()` context, session.run() marshaling) very plausibly dominates over the matmul itself — this experiment doesn't yet isolate how much of the 6-13x is "dispatch overhead" vs. "our C++ path skips work ONNX Runtime's generic graph executor still does." That decomposition is a good next experiment, not something to claim here.
- **RQ4 (memory/dependency)**: peak RSS is ~38-41x smaller (5.75MB vs 217-238MB), and on-disk dependency footprint is 78-536MB (Python) vs. 0.04MB (one static-ish binary). This is a real, measured, reproducible number — one of the more defensible claims from this experiment.
- **Caveats that limit generalization**: one model, one (tiny) size, one machine (Apple Silicon, no CUDA comparison possible per environment.md), one run. No multi-threading, no batching beyond size 1, no larger model tested. This experiment establishes the methodology and gets an honest first data point — it does not establish that these ratios hold for larger/different models, and shouldn't be cited as if it does.

## 4. What this experiment does and doesn't prove

**Establishes**: the equivalence-measurement harness works and produces a real, tight, explainable tolerance; the benchmark harness produces fair, reproducible, machine-documented numbers; for this trivial model, native C++ wins decisively on cold start and dependency footprint, and meaningfully (not overwhelmingly) on warm per-call latency.

**Does not establish**: that this holds for larger models, batched inference, multi-threaded serving, or non-Apple-Silicon hardware; that the warm-latency gap is due to "C++ vs. Python" rather than "hand-written minimal dispatch vs. general-purpose graph executor" (ONNX Runtime is also C++ under the hood — the gap there is architectural, not language-based, and should be described that way).

## 5. Suggested next experiment

Repeat this exact harness on a model 10-100x larger (still non-LLM — e.g. a small CNN or a 2-3 layer transformer block) to see whether the warm-latency ratio holds, shrinks, or grows as actual compute time starts to dominate over dispatch overhead — this directly tests whether the "native wins most for small/single-request inference" hypothesis (RQ6) is really about model size or was an artifact of picking an unrealistically tiny first model.
