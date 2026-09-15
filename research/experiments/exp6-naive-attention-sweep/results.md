# Experiment 6 — Hand-Written Attention: A Negative Result

**Prediction from [exp5](../exp5-gemm-dispatch-overhead/results.md)**: replacing transformer.cpp's 8 per-head `cblas_sgemm` calls with a single hand-written (BLAS-free) multi-head attention pass should close some of the gap with ONNX Runtime, since exp5 showed BLAS's fixed per-call dispatch cost dominates at this matrix size.
**What we built**: [`multi_head_attention_naive()`](../../../native/common.hpp) — plain nested loops computing all heads' scores and context directly, zero `cblas_sgemm` calls, replacing the per-head loop in [transformer.cpp](../../../native/transformer.cpp). Equivalence reverified first (max abs error 1.43e-6, still fp32-scale — a valid reimplementation, not a bug). Then reran exp3/exp4's identical sweep.

## Result: strictly worse at every size tested

| d_model | exp4 (BLAS per-head) native p50 | exp6 (naive) native p50 | exp4 native/onnx | exp6 native/onnx |
|---:|---:|---:|---:|---:|
| 16 | 0.02121 ms | 0.02415 ms | 1.25x | 1.12x |
| 32 | 0.03250 ms | 0.03629 ms | 0.99x | 0.94x |
| 64 | 0.05081 ms | 0.06913 ms | 1.08x | 0.78x |
| 128 | 0.09538 ms | 0.12358 ms | 0.94x | 0.78x |
| 192 | 0.13017 ms | 0.18923 ms | 1.05x | 0.77x |
| 256 | 0.20394 ms | 0.26867 ms | 0.89x | **0.62x** |

Full data: [results.json](results.json). **The naive implementation is slower than the BLAS-per-head version at every single size, and the gap widens as `d_model` grows** — the opposite of the predicted improvement, and worse in the direction that matters most (larger, more realistic model sizes).

## Why the prediction failed — a methodological gap between exp5 and this experiment

exp5 controlled for kernel quality and varied only call count: "k separate BLAS calls" vs. "1 batched BLAS call," both using Accelerate's optimized kernel internally. **exp6 did not have that option** — Accelerate has no batched-GEMM primitive (`cblas_?gemm_batch`), so "fewer calls" here meant "hand-written scalar loops" instead, which changes two variables at once: call count *and* kernel quality. The naive loops get none of Accelerate's AMX-coprocessor use, SIMD vectorization, or cache-blocking — and that lost throughput apparently outweighs the dispatch-overhead savings almost immediately, growing worse as `d_head` (and thus real per-head compute) grows with `d_model`. exp5's measured "3.77x separate-call penalty at k=28" was real, but it was a penalty on top of BLAS's own high throughput — not license to assume any BLAS-free alternative would net out ahead once compute stops being negligible.

**The honest conclusion: dispatch overhead is real (exp5 established that), but it does not mean "avoid BLAS" is the fix — it means BLAS's own high per-FLOP efficiency was, in this case, worth more than its per-call overhead cost, for every size actually tested.** This experiment isolates that BLAS's compute efficiency (AMX, SIMD, cache blocking) outweighs its dispatch cost here, not that the dispatch cost doesn't exist.

## Action taken

Reverted `transformer.cpp`'s attention back to the per-head `cblas_sgemm` implementation (exp4's version) — it's strictly faster than this experiment's alternative at every tested size, so there's no reason to leave a known-worse implementation as the current code. `multi_head_attention_naive()` stays in [common.hpp](../../../native/common.hpp), documented as a tested-and-rejected alternative, not deleted — the negative result and the reasoning for it are worth keeping visible for anyone who reaches for "just hand-write it" as the fix next time.

## What would actually need to be true for the original prediction to hold

A *properly vectorized* batched attention kernel — not a naive one — closing enough of the compute-efficiency gap with Accelerate's kernels that the remaining dispatch-overhead savings show through net positive. That's a real hardware-level optimization task (explicit SIMD/NEON intrinsics, or restructuring for the compiler's auto-vectorizer, then checking the generated assembly) — not attempted here, and a large enough undertaking that it should be scoped as its own deliberate experiment rather than assumed as a quick fix.

## Next experiment

Two independent directions worth separating rather than conflating (as this experiment accidentally did): (1) verify whether the compiler actually auto-vectorized `multi_head_attention_naive()`'s inner loops at `-O3` (check the generated assembly or a vectorization report) — if it didn't, a version with explicit SIMD hints might change this result; (2) step back from micro-optimizing attention and instead ask whether the entire premise needs revisiting — given exp2-exp6's cumulative finding that native C++ has not beaten ONNX Runtime CPU on warm latency for *any* transformer-shaped workload tested so far, is the project's actual defensible advantage (per research/gap-analysis.md) really cold-start and dependency footprint, not warm-latency, for this class of model? That's a bigger, more honest question than the next microbenchmark, and worth confronting directly before sinking more effort into kernel-level tuning.
