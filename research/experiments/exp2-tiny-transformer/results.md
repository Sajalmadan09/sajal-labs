# Experiment 2 — Tiny Transformer Block: PyTorch vs. Native C++ (Accelerate)

**Model**: one post-norm Transformer encoder block (multi-head self-attention + residual + LayerNorm, GELU feed-forward + residual + LayerNorm). `seq_len=32, d_model=256, n_heads=4, d_ff=1024`. Random weights (seed 0). ~25.7M MACs/forward pass — **~2,712x more compute than [exp1](../exp1-tiny-mlp/results.md)'s tiny MLP** (9,472 MACs).
**Hardware/compiler/Python**: identical to exp1 — see [environment.md](../../environment.md).
**Methodology**: identical to exp1 — batch size 1, single-threaded, 50 warmup + 500 measured iterations, same percentile method both languages, peak RSS via `/usr/bin/time -l`. Code: [tiny_transformer.py](../../../python/tiny_transformer.py), [transformer.cpp](../../../native/transformer.cpp). Raw numbers: [results.json](results.json).

## 1. Numerical equivalence

| Metric | exp1 (MLP) | exp2 (transformer) |
|---|---:|---:|
| Max absolute error | 4.47e-08 | 1.67e-06 |
| Mean absolute error | 6.17e-09 | 1.37e-07 |
| RMSE | 9.11e-09 | 1.88e-07 |
| Mean cosine similarity | 1.0 | 1.0 |

Error grew ~37x from exp1 to exp2, but compute grew ~2,712x — error accumulated much more slowly than FLOP count, consistent with fp32 rounding error growing roughly with the *depth* of sequential operations (6 matmuls + 2 LayerNorms + GELU vs. 2 matmuls) rather than with total FLOPs. Still comfortably fp32-scale — no sign of a LayerNorm-eps or GELU-formula mismatch. See [equivalence_result.json](../../../artifacts/transformer/equivalence_result.json).

## 2. Benchmark results

| Implementation | Cold start (ms) | Warm p50 (ms) | Warm p99 (ms) | Peak RSS (MB) |
|---|---:|---:|---:|---:|
| **Native C++ (Accelerate)** | 3.57 | 0.1856 | 0.196 | 10.73 |
| PyTorch eager | 711.05 | 0.2656 | 0.328 | 229.02 |
| ONNX Runtime (CPU EP) | 532.75 | 0.1843 | 0.214 | 253.28 |

## 3. The headline finding: the warm-latency advantage nearly disappears at this compute scale

| | exp1 warm p50 ratio (native vs. baseline) | exp2 warm p50 ratio |
|---|---:|---:|
| vs. PyTorch eager | 12.6x faster | 1.43x faster |
| vs. ONNX Runtime CPU | 6.2x faster | **~1.0x — statistical tie** (native p50 0.1856ms vs. ONNX Runtime 0.1843ms; ONNX Runtime is marginally *faster* on p50, native has a tighter p99 tail) |

This is exactly the effect flagged as untested in exp1's results: at exp1's trivial size, warm latency was dominated by *dispatch overhead* (Python object allocation, `torch.no_grad()` context, `session.run()` marshaling) — and our hand-written C++ skips that overhead entirely. At exp2's larger size, warm latency is dominated by *matmul compute* — and ONNX Runtime's CPU execution provider is calling into its own well-optimized kernels (MLAS, likely also using Accelerate-equivalent paths on Apple Silicon) just as effectively as our code calls into Accelerate directly. Once compute dominates over dispatch, "hand-written minimal C++" stops being meaningfully faster than "mature C++ runtime with a generic graph executor," because both bottleneck on the same underlying BLAS work.

**What did NOT change with model size:**
- **Cold start** stayed a ~150-200x gap (3.57ms vs. 533-711ms) — confirms this is a Python-process-startup effect (import cost of large C-extension libraries), independent of model size, exactly as exp1 hypothesized.
- **Peak RSS** stayed a large gap (~21-24x smaller), though the ratio shrank somewhat (was ~38-41x in exp1) because native's own footprint grew with the model's ~3.1MB of weights while Python's ~230-253MB is dominated by library-loading overhead that doesn't grow with model size.

## 4. Revised interpretation of the project's central hypothesis

Exp1 alone could have been read as "native C++ is 6-13x faster at inference" — exp2 shows that claim doesn't hold as compute scales, and would have been an overclaim if generalized from one tiny model. The more accurate, defensible claim after two data points:

- **Cold-start and dependency-footprint advantages are large and appear model-size-independent** — driven by avoiding Python/framework process startup, not by any property of the inference computation itself. This matters most for serverless/CLI/one-shot/edge invocation (RQ6), and is Sajal Labs' most defensible benchmark claim so far.
- **Warm per-call latency advantage shrinks toward zero as the model's actual compute grows relative to framework dispatch overhead**, because the matmul-heavy work is delegated to the same class of optimized BLAS kernel on both sides. This means "native C++" is not, by itself, a source of inference speedup for compute-bound models — the speedup in exp1 was really "less dispatch overhead for a dispatch-bound workload," which is a narrower and more honest claim than "native is faster."
- This reframes what a compiler-based Sajal Labs should actually promise: **the win is in removing framework/process overhead and shrinking dependency footprint, not in beating mature runtimes at raw compute-bound matmul throughput** — which lines up with research/gap-analysis.md's conclusion that hand-tuned kernels should be delegated to Accelerate/BLAS rather than reimplemented, and reframes RQ5 ("when does native provide a meaningful advantage") with real evidence: dispatch-bound (tiny/single-request) workloads, not compute-bound ones.

## 5. Caveats

Two data points on a curve, not a full curve — we don't yet know *where* between exp1's size and exp2's size the crossover from "dispatch-bound" to "compute-bound" happens, only that it happens somewhere in between. Single-threaded only; ONNX Runtime's advantage in a multi-threaded/batched setting is untested. Still one machine (Apple Silicon, no CUDA).

## 6. Suggested next experiment

Run the same harness at 2-3 intermediate sizes between exp1 and exp2 (e.g. d_model=32, 64, 128) to actually locate the crossover point, rather than only having the two endpoints — that turns "the advantage shrinks somewhere in between" into a measured curve.
