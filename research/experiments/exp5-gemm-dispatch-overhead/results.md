# Experiment 5 — Isolating cblas_sgemm's Per-Call Dispatch Overhead

**Question from [exp4](../exp4-buffer-reuse-sweep/results.md)**: transformer.cpp makes 14 separate `cblas_sgemm` calls per forward pass (4 projections + 8 per-head attention matmuls + 2 FFN linears) vs. mlp.cpp's 2 — does Accelerate have a fixed per-call dispatch cost large enough, relative to a small matmul's actual compute, to explain why the transformer lost its advantage over ONNX Runtime at every size tested? No PyTorch/ONNX comparison in this experiment — purely measuring Accelerate's own call-dispatch cost in isolation. Code: [gemm_bench.cpp](../../../native/gemm_bench.cpp). Raw data: [size_sweep.csv](size_sweep.csv), [call_count.csv](call_count.csv). Same hardware/compiler as prior experiments — [environment.md](../../environment.md). 50 warmup + 200 measured calls per data point.

## Part A: latency vs. matrix size (square M=K=N)

| Size | p50 latency |
|---:|---:|
| 1 | 0.00008 ms (80 ns) |
| 2 | 0.00008 ms |
| 4 | 0.00008 ms |
| 8 | 0.00017 ms |
| 16 | 0.00046 ms |
| 32 | 0.00071 ms |
| 64 | 0.00300 ms |
| 128 | 0.02256 ms |
| 256 | 0.02285 ms |
| 512 | 0.17404 ms |

At the smallest sizes (1-4), latency is flat at ~80ns — real compute for a 4x4x4 matmul (128 FLOPs) is a few nanoseconds at most, so this ~80ns is (approximately) **the fixed per-call dispatch floor**: thread-pool/work-scheduling setup inside Accelerate that a tiny matmul pays regardless of how little actual work it does. There's also a sharp jump between size 64 (3μs) and 128 (22.6μs) — plausibly a threading-strategy threshold inside Accelerate (e.g. switching to multi-threaded execution, which itself carries setup cost) — noted as a caveat, not investigated further here.

## Part B: same total work, fewer vs. more calls

Two shapes chosen to match real workloads from exp2-4: `attn_head` (M=32,K=64,N=32 — one attention head's score/context matmul shape) and `linear_proj` (M=32,K=256,N=256 — a full linear projection at d_model=256). For each, `k` separate calls of width N are compared against 1 batched call of width `k×N` — **identical total FLOPs either way, only the call count differs.**

| Shape | k | separate (ms) | batched (ms) | separate/batched |
|---|---:|---:|---:|---:|
| attn_head | 1 | 0.00038 | 0.00033 | 1.13x |
| attn_head | 2 | 0.00063 | 0.00050 | 1.25x |
| attn_head | 4 | 0.00117 | 0.00052 | **2.24x** |
| attn_head | 8 | 0.00242 | 0.00079 | **3.05x** |
| attn_head | 14 | 0.00408 | 0.00121 | **3.38x** |
| attn_head | 28 | 0.00817 | 0.00217 | **3.77x** |
| linear_proj | 1 | 0.00258 | 0.00254 | 1.02x |
| linear_proj | 2 | 0.00517 | 0.00496 | 1.04x |
| linear_proj | 4 | 0.01200 | 0.01290 | 0.93x |
| linear_proj | 8 | 0.02367 | 0.02483 | 0.95x |
| linear_proj | 14 | 0.04231 | 0.03725 | 1.14x |
| linear_proj | 28 | 0.08798 | 0.09888 | 0.89x |

## The hypothesis is confirmed — but only in the regime that matters

**At attention-head scale, splitting identical work into more calls really does cost more, and the gap widens with call count**: 4 separate calls already cost 2.24x a single batched call of the same total work; by 28 calls it's 3.77x. This is exactly the mechanism suspected in exp4 — and it's exactly the shape transformer.cpp's per-head attention loop uses (8 of its 14 calls per forward pass are this `attn_head`-shaped case: `n_heads × 2` for scores + context).

**At linear-projection scale, batching makes no consistent difference** (ratios hover at 0.89-1.14x, essentially noise) — at this larger matrix size, actual compute already dominates over per-call dispatch cost, so call count stops mattering. This matches Part A's finding that the fixed-cost floor becomes negligible once matrix size grows past roughly the 64-128 range.

**This directly explains exp3/exp4's result**: transformer.cpp's 8 small per-head attention calls are exactly the regime where call-count overhead is real and measurable (up to ~3.8x at this experiment's largest k), while its 6 larger linear-layer calls are in the regime where it doesn't matter. The fixed-size, per-head structure of the naive attention loop is a genuine, now-measured inefficiency — not the buffer allocation exp4 ruled out.

## Caveats

This measures `cblas_sgemm` in isolation with synthetic data, not the full transformer forward pass — it establishes the mechanism plausibly exists and roughly how large it is, not that fixing it will fully close the gap with ONNX Runtime (ONNX Runtime may also batch/fuse internally, or have other advantages independent of this one). The size-128 latency jump in Part A is unexplained and could itself be a confound worth isolating separately later.

## Next experiment

Rewrite `transformer.cpp`'s per-head attention loop to use batched GEMM calls across all heads (matching this experiment's "batched" construction — one wider call instead of `n_heads` separate ones) instead of the current per-head loop, then rerun exp3/exp4's size sweep to see how much of the gap with ONNX Runtime actually closes. That turns this microbenchmark's finding into a testable prediction against the real model.
