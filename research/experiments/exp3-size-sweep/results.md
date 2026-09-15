# Experiment 3 — Size Sweep: Locating the Crossover

**Goal**: exp2 found native C++'s warm-latency edge over ONNX Runtime CPU shrank from exp1's 6.2x to ~1.0x at 2,712x more compute — but with only two data points, we didn't know *where* between them the crossover happens, or whether it's really about raw compute at all. This sweeps `d_model ∈ {16, 32, 64, 128, 192, 256}` (same architecture as exp2, `seq_len=32`, `n_heads=4`, `d_ff=4×d_model`) to find out.
**Hardware/compiler/Python/methodology**: identical to exp1/exp2 — see [environment.md](../../environment.md). Code: [run_sweep.py](../../../python/run_sweep.py) (reuses [export_transformer.py](../../../python/export_transformer.py)'s `export()`, the parameterized bench/compare scripts, and [transformer.cpp](../../../native/transformer.cpp) unchanged — it already reads shape from `shapes.txt` at runtime). Raw numbers: [results.json](results.json).

## Results

| d_model | MACs/pass | max abs err | native p50 (ms) | onnx p50 (ms) | native vs. onnx | native vs. pytorch | native peak RSS (MB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 131,072 | 7.2e-07 | 0.0233 | 0.0239 | **1.03x** | 2.69x | 5.88 |
| 32 | 458,752 | 7.2e-07 | 0.0307 | 0.0339 | **1.11x** | 2.37x | 6.00 |
| 64 | 1,703,936 | 9.5e-07 | 0.0557 | 0.0507 | **0.91x** | 1.54x | 6.24 |
| 128 | 6,553,600 | 1.19e-06 | 0.0972 | 0.0890 | **0.92x** | 1.05x | 7.08 |
| 192 | 14,548,992 | 1.43e-06 | 0.1480 | 0.1315 | **0.89x** | 1.23x | 8.57 |
| 256 (=exp2) | 25,690,112 | 1.67e-06 | 0.2094 | 0.1785 | **0.85x** | 1.12x | 11.03 |

("native vs. X" = X's p50 ÷ native's p50, so >1 means native is faster.)

## The finding this sweep actually surfaces — and it's not the one expected

The plan was to find *where between exp1 and exp2* native's edge over ONNX Runtime crosses from >1x to ~1x. **That's not what happened.** Native vs. ONNX Runtime is already at ~1.0x at `d_model=16` — the *smallest* transformer tested, with only 131K MACs/pass, less compute than exp1's plain MLP's 9,472 MACs would suggest is needed to reach parity if "total FLOPs" were the driver. By `d_model=64` onward, ONNX Runtime is consistently *faster* than our native code (ratio <1).

**The crossover against ONNX Runtime isn't inside this sweep's range at all — it happened between exp1's architecture (plain MLP) and any transformer architecture, regardless of transformer size.** That's a different, more specific claim than exp2's "crossover happens somewhere as compute scales," and it changes the likely explanation.

Against PyTorch eager, native keeps a real edge across the whole sweep (2.69x down to ~1.1-1.2x) — noisier at the low end but never crossing to <1x — which is a separate comparison from the ONNX Runtime one and shouldn't be conflated with it.

## Why: a likely confound worth naming, not hiding

[transformer.cpp](../../../native/transformer.cpp)'s `forward()` allocates roughly ten `std::vector`s (Q, K, V, ctx, scores, attn_out, x1, ff_hidden, ff_out, x2) — fresh heap allocations on *every call* — versus [mlp.cpp](../../../native/mlp.cpp)'s two (H, Y). ONNX Runtime's session pre-plans and reuses an internal memory arena across `session.run()` calls instead of allocating fresh buffers each time. **It's plausible that what this sweep is actually measuring, at least partly, is "heap allocation count per call" rather than "compute intensity"** — the transformer's *operator count* (6 linears + per-head attention + 2 LayerNorms + GELU + residuals ≈ 15+ steps) jumped discontinuously from exp1's 2 linears the moment we moved to *any* transformer size, while FLOPs scaled continuously. That would explain why the crossover already happened at the smallest transformer rather than showing up gradually within this sweep's range.

This wasn't controlled for and shouldn't be presented as settled — it's the most concrete, checkable next step.

## What holds up regardless

- **Numerical equivalence held at every size** — max abs error grows slowly and smoothly with `d_model` (7.2e-7 to 1.67e-6), no discontinuity, no sign of a bug at any tested size.
- **Peak RSS grows with weight count as expected** (5.88MB → 11.03MB) but stays far below PyTorch/ONNX Runtime's ~217-253MB floor (dominated by library loading, not model size) at every size tested — the memory-footprint and cold-start advantages from exp1/exp2 look robust across this whole range, unlike the warm-latency comparison against ONNX Runtime.

## Revised interpretation

exp2's framing ("dispatch-bound vs. compute-bound, crossing over as FLOPs scale") was too simple. What actually seems to matter more, based on this sweep, is **implementation-level allocation/buffer-reuse discipline**, not raw FLOP count — and our hand-written C++ currently loses that comparison to ONNX Runtime's mature memory arena the moment the model has more than a couple of operators, regardless of how small those operators are. exp1's 6x advantage may have been specific to a 2-operator model with minimal allocation overhead on our side, not representative of "small models" in general.

## Next experiment

Before sweeping anything else: **fix the allocation confound and re-measure.** Pre-allocate `transformer.cpp`'s intermediate buffers once (outside the warmup/timing loop, reused across all 500 iterations) instead of allocating fresh `std::vector`s per `forward()` call, then rerun this same sweep. If native pulls back ahead of ONNX Runtime once allocation overhead is removed, that confirms allocation was the real variable, not compute — and tells us buffer-reuse discipline, not "native vs. framework," is the thing that actually matters for Sajal Labs' eventual compiler to get right.
