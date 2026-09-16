# Experiment 22 — The Compiler vs. exp2's Own Baseline

**Why this experiment**: exp15-21 validated the compiler exclusively on correctness (bit-identical output), and exclusively at toy scale (`d=16`, `seq_len=8`) — never against a real benchmark number, and never at a size anyone actually measured performance at. exp21 closed the correctness gap (exp2's real architecture, composed for free). This closes the other gap: does the **auto-generated** code cost anything — in cold-start, warm-loop latency, memory, or binary size — compared to the **hand-written** code this whole project's numbers (exp2-13) were built on?

**The result**: no. Cold-invocation, warm p50, peak RSS, and binary size are all statistically indistinguishable between `sajal_compile.py`'s generated code and exp2's original `transformer.cpp`, run on the exact same architecture, weights, and input.

## Setup: testing against exp2 itself, not a re-derived equivalent

[python/export_exp2_scale_test.py](../../../python/export_exp2_scale_test.py) imports `TinyTransformerBlock` directly from [python/tiny_transformer.py](../../../python/tiny_transformer.py) — exp2's actual, unmodified model class, at its actual benchmarked size (`seq_len=32, d_model=256, n_heads=4, d_ff=1024`) — rather than re-implementing an equivalent module at this scale. It writes the SAME weights and input in two conventions: `model.onnx` for `sajal_compile.py`, and `shapes.txt` + `*_weight.bin`/`*_bias.bin` in `transformer_model.hpp`'s exact naming convention, so `native/transformer.cpp` (exp2's own driver, unchanged since exp2) loads and runs the identical model.

## A real gap, found by testing against the real thing instead of a synthetic model

Compiling this graph failed on the first attempt: `error: op 'Reshape' is not supported...`. Inspecting the ONNX graph (same discipline as every prior experiment) showed why — `tiny_transformer.py`'s attention scaling is `scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_head)`, a **`Div`** node. Every synthetic compiler test model so far (exp18/20/21) wrote `scores = (q @ k.transpose(-2,-1)) * self.scale` — a **`Mul`** node. Mathematically identical, but two different ONNX ops, and the pattern matchers only recognized `Mul`. This is exactly the kind of gap synthetic test models can hide and a real model exposes.

Fixed by generalizing both `try_match_self_attention` and `try_match_multihead_attention` to a shared `try_match_scale` helper that accepts either `Mul(x, c)` or `Div(x, c)` (the latter as scale `1/c`) — a few lines, not a redesign. Regression-tested: all 8 prior compiler targets still bit-identical afterward.

## Also added: a `bench` mode to the compiler's codegen

`sajal_compile.py`'s generated code only had `run`/`once` modes — no way to measure warm-loop latency with the project's own established methodology (50 warmup + 500 iterations, percentiles via `common.hpp`'s `percentile()`). Added `bench_mode` to `generate_cpp()`'s template, mirroring `transformer.cpp`'s exactly (same JSON print shape, `impl: "compiled_cpp"`). Also fixed `once_mode` to use the full sequence length (read from `test_config.txt`) rather than a hardcoded single row for attention-containing models — one row can't attend to anything, and more importantly wouldn't be a representative "one inference" for a real single-shot invocation. Non-attention models (exp1/9) keep `n=1`, unchanged, matching `mlp.cpp`/`gender_predict.cpp`'s own convention. Neither change touches `run_mode`, so exp1-21's already-validated bit-identical outputs are untouched (reverified below).

## Validation

**Regression, all 8 prior compiler targets** (exp1, exp9, exp16-21) recompiled from scratch after both changes (`Div` support, `bench`/`once` fixes) — still bit-identical.

**Equivalence at real scale**: compiled output is **bit-identical** to `native/transformer.cpp`'s output (both loading the exact same weights) — `cmp` reports no difference. PyTorch equivalence: max abs error `1.19e-06`, matching exp2's own published number (`1.67e-06`) in order of magnitude (different only because of different random weights/seed).

## Benchmark: compiled vs. hand-written, same methodology as exp2/exp7

| Metric | exp2 hand-written (`transformer.cpp`) | Compiled (`sajal_compile.py` output) |
|---|---:|---:|
| Cold invocation, wall-clock, p50 (30 trials) | 4.610 ms | 4.626 ms |
| Cold invocation, wall-clock, p95 | 5.642 ms | 5.654 ms |
| Warm loop, p50 (50 warmup + 500 iters) | 0.216 ms | 0.222 ms |
| Peak RSS | 10.85 MB | 10.80 MB |
| Binary size | 58.8 KB | 58.0 KB |

Cold-invocation methodology matches exp7/8 exactly ([python/run_compiler_cold_invocation_bench.py](../../../python/run_compiler_cold_invocation_bench.py): `bench_common.time_invocation`, genuinely fresh OS process per trial, interleaved round-robin so neither binary's page cache gets an unfair head start). Raw numbers: [results.json](results.json).

Every metric is within noise of the other — none of these differences would survive a second run in a different order. The compiler is not slower, not larger, not more memory-hungry than the code a human wrote for this exact architecture.

## Interpretation

exp15-21 proved the compiler produces **correct** code. This proves it produces code with **no performance cost** relative to the hand-written alternative it's meant to replace — the natural remaining doubt about any "compile this for me" tool (that convenience costs something) doesn't hold here, at least at this one architecture and scale. This makes sense given *why* it costs nothing: the compiler doesn't invent its own numerical kernels — it emits calls into the exact same `common.hpp`/`cblas_sgemm` primitives a human would write by hand (documented since exp15's docstring), so "auto-generated" here means "auto-assembled from the same building blocks," not "a different, less-optimized code path."

This also means the project's actual value proposition — cold-start and footprint advantage over Python/ONNX Runtime, established in exp7-13 — now demonstrably transfers to compiler output, not just to hand-ported code. A real model exported from PyTorch, run through `sajal_compile.py` with zero model-specific code written, gets the same ~150-200x cold-start advantage exp2 already measured against PyTorch/ONNX Runtime (not re-measured here, since that comparison doesn't involve the compiler and exp2's numbers already answer it — re-running it would only be useful to confirm environment drift, not to test anything new about the compiler).

## Caveats

One architecture, one size, one machine (same caveat every prior benchmarking experiment carries). The `Div`-vs-`Mul` fix widens what the compiler accepts, but is still a fixed, recognized pattern — a model using some third mathematically-equivalent scaling formulation (e.g. `Mul` by a `Constant` produced by a `Reciprocal`+`Sqrt` chain instead of a precomputed scalar) would still fail cleanly rather than being silently handled. `bench`/`once` mode's `n=1` vs. `n=full-sequence` branching is a simple attention-presence check, not a general "what counts as one inference" solver — fine for every model this compiler currently accepts, since none mixes an attention op with a batch-of-independent-rows convention.

## Next experiment

The compiler has now cleared both bars this project cares about: correct (exp15-21) and free (this experiment). Candidates from here: stack multiple encoder blocks (does the IR/codegen handle N repeated blocks, still zero new code?); widen the accepted op set to decomposed (opset<20) GELU, since real models exported with older opsets would currently fail cleanly rather than compile; or move toward publishing the compiler itself as a usable tool (a CLI wrapper, docs) now that both correctness and performance claims are backed by evidence rather than assumption.
