# Experiment 4 — Does Buffer Reuse Explain exp3's Result?

**Hypothesis under test** (from [exp3](../exp3-size-sweep/results.md)): native C++ lost its edge over ONNX Runtime at every transformer size, even the smallest (131K MACs). The leading suspect was `transformer.cpp` allocating ~10 fresh `std::vector`s per `forward()` call versus ONNX Runtime's reused memory arena — a confound distinct from actual compute.
**Fix applied**: [transformer.cpp](../../../native/transformer.cpp) now allocates all scratch buffers (`Q, K, V, ctx, scores, attn_out, x1, ff_hidden, ff_out, x2`) once in `allocate_buffers()`, called right after weights load. `forward()` writes into these reused buffers and returns a `const&` instead of allocating and returning by value. Rebuilt, reverified equivalence unchanged (max abs error still 1.67e-06 at d_model=256 — this was a memory-management change, not a math change). Otherwise identical setup to exp3 — same six `d_model` values, same [run_sweep.py](../../../python/run_sweep.py) (now taking an experiment-name argument so this run writes here instead of overwriting exp3).

## Result: the hypothesis is not confirmed

| d_model | exp3 native p50 (ms) | exp4 native p50 (ms) | exp3 native/onnx | exp4 native/onnx | exp3 RSS (MB) | exp4 RSS (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.02329 | 0.02121 | 1.03x | 1.25x | 5.88 | 5.83 |
| 32 | 0.03071 | 0.03250 | 1.11x | 0.99x | 6.00 | 6.00 |
| 64 | 0.05573 | 0.05081 | 0.91x | 1.08x | 6.24 | 6.21 |
| 128 | 0.09721 | 0.09538 | 0.92x | 0.94x | 7.08 | 7.14 |
| 192 | 0.14800 | 0.13017 | 0.89x | 1.05x | 8.57 | 8.59 |
| 256 | 0.20938 | 0.20394 | 0.85x | 0.89x | 11.03 | 10.88 |

Full data: [results.json](results.json).

**Removing per-call heap allocation produced a small, noisy improvement (roughly +0.1-0.15x on the native/onnx ratio, within run-to-run variance for a few of these points) — not the decisive recovery of exp1's 6x advantage the hypothesis predicted.** Peak RSS barely moved either (as expected in hindsight: RSS in a `bench` process is dominated by weight storage and the OS/Accelerate library footprint, not by a handful of small reused-vs-reallocated scratch buffers). Native and ONNX Runtime remain roughly at parity (0.89x-1.25x) across the whole size range, both before and after the fix.

**Conclusion: allocation overhead was not the primary driver of exp3's result.** It's ruled out, not confirmed — a real, useful negative result, and worth stating plainly rather than searching for a way to call the noisy +0.1x "the explanation."

## A better-supported alternative, visible in the code itself

`transformer.cpp`'s `forward()` makes **14 separate `cblas_sgemm` calls** per forward pass (4 linear projections + 2 per-head score/context matmuls × 4 heads + 2 FFN linears = 4+8+2=14), versus [mlp.cpp](../../../native/mlp.cpp)'s **2**. Each `cblas_sgemm` call has its own fixed dispatch cost inside Accelerate — thread-pool/work-scheduling setup that doesn't shrink just because the matrix is small — independent of whether the *caller's* surrounding buffer is fresh or reused. If that per-call fixed cost is large relative to a small matmul's actual compute, then **operator count** (how many separate kernel calls a forward pass makes), not allocation count or raw FLOPs, is the more likely variable behind exp3's result. ONNX Runtime may also be making fewer, larger, or better-fused kernel calls for the same graph (bias-add fused into GEMM, multiple ops fused into one kernel — exactly the operator-fusion technique research/papers.md §4 covers), which would explain why it holds parity with native even at small sizes where native's per-op FLOPs are tiny.

This also reconnects to a gap named in [research/papers.md](../../papers.md) §4: none of the fusion literature (DNNFusion, TASO) measures whether fused output stays numerically equivalent to the unfused reference — if Sajal Labs' compiler ever fuses operators to cut call count, that equivalence question becomes directly relevant, not just theoretical.

## Caveats

This experiment rules something out; it doesn't yet prove the operator-count explanation, only motivates it as more plausible than the allocation hypothesis it replaces. The "14 GEMM calls" count is exact from reading the code, but "per-call dispatch cost is the bottleneck" is not yet measured — it's the next thing to check, not a conclusion.

## Next experiment

Directly measure Accelerate's per-call `cblas_sgemm` fixed cost: benchmark N tiny (e.g. 4x4) matmuls done as N separate calls vs. one batched/larger call doing equivalent total work, isolating dispatch overhead from compute. If fixed per-call cost turns out to be, say, several microseconds regardless of matrix size, that would explain why a 14-call forward pass loses to a 2-call one even though both are tiny in FLOPs — and would point Sajal Labs' compiler toward operator fusion (reducing kernel-call count) as the actual lever, rather than "write more native code" or "allocate less."
