# Experiment 31 — Real GPT-2's Full 12-Layer Stack

**Why this experiment**: exp30 closed the loop on compiling real GPT-2 directly, but only tested 2 of its 12 real layers — enough to prove the mechanism, not enough to prove depth doesn't matter. exp23 found a real bug the moment synthetic blocks were stacked past one instance (a global shape-scan matcher that silently merged branches from different attention instances). This asks the same question of the now-complete real-GPT-2 line: does real depth surface anything new, now that every individual per-layer capability (combined QKV via `Split`, the additive causal mask, `tanh`-GELU, `Conv1D`'s `transB=0`/reshape-wrapping) is already closed?

**The result**: no. Real GPT-2's **entire** 12-layer decoder stack — all 900 ONNX nodes, traced directly with zero rewiring — compiles with **zero changes to `sajal_compile.py`**, in about 4.5 seconds, to a 714-line, 170KB generated binary. This is the cleanest result in the whole GPT-2 arc (exp28-31): nothing to fix, nothing to route around.

## What changed from exp30: nothing but the layer count

[python/export_real_gpt2_full_stack_test.py](../../../python/export_real_gpt2_full_stack_test.py) is exp30's [export_real_gpt2_direct_test.py](../../../python/export_real_gpt2_direct_test.py) with `model.h[:2]` replaced by `model.h` (all 12 real layers, `model.config.n_layer`) — same real weights, same direct trace, same zero rewiring. No new op, no new pattern, no new matcher logic. `git status` on `sajal_compile.py` after this experiment shows no diff at all.

## Validation

**Regression, all 20 prior compiler targets** (exp1, exp9, exp16-30) recompiled from scratch — unaffected, trivially, since the compiler itself didn't change.

**Full 12-layer real GPT-2 stack**: compiles cleanly. PyTorch equivalence vs. calling all 12 real layers directly (`model.h[i](x)` in a loop, no ONNX involved): max abs error `1.83e-04` — close to exp30's 2-layer number (`1.22e-04`), growing gently with depth exactly as every prior experiment's error-accumulation pattern predicts (exp2, exp11, exp19, exp23), not exploding.

**Not bit-identical against a hand-written reference** ([native/real_gpt2_full_stack_reference.cpp](../../../native/real_gpt2_full_stack_reference.cpp), a parameterized `GPT2Block` looped 12 times) — `2.21e-04` max diff, for the exact same reason exp26 and exp30 already documented: the hand-written reference uses a single post-matmul attention scale, while the real compiled trace uses GPT-2's actual split pre-matmul scaling (combining two independently-rounded `d_head**-0.25` factors). That divergence compounds gently across 12 layers instead of 2 (`2.21e-04` vs. exp30's `1.14e-05` for the same comparison at 2 layers) — consistent with the same benign floating-point non-associativity, not a new phenomenon requiring explanation. The PyTorch-equivalence number above is the authoritative correctness check here, and it stayed tight.

## Interpretation

exp23's fix — making the multi-head attention matcher purely tensor-flow-traced rather than relying on a global shape scan — was motivated by, and validated against, synthetic stacked blocks. This experiment confirms that same fix generalizes to what it was actually meant for: a real model's real depth, with every other per-layer quirk (`Split`, the additive mask, `tanh`-GELU) also in play simultaneously across 12 repetitions. Twelve independent attention instances, twelve independent causal masks, twelve independent combined-QKV buffers — all correctly scoped to their own layer, none bleeding into another's.

This also closes out the "does depth alone break anything" question the GPT-2 arc opened at exp30's own suggestion. Combined with exp28-30's per-capability closures, the compiler now handles real GPT-2 (small) in its entirety at the decoder-stack level — not a hand-picked 2-layer sample of it.

## Caveats

Same scope note carried since exp25: the decoder-block stack alone (token/position embeddings and `lm_head` are outside this compiler's op set — `Gather`-based embedding lookup was never attempted). One sentence, one sequence length, one machine.

## Next experiment

The natural remaining direction is embeddings: token/position lookup via `Gather`, which would let the compiler take raw token IDs in and finish the job started at exp10/14 (native tokenization) all the way through a compiled forward pass, rather than requiring pre-computed embeddings as input. Alternatively, benchmark the full 12-layer compiled stack's cold-start/footprint against GPT-2's own PyTorch baseline, the way exp22 did for exp2 — a real end-to-end performance claim for a real, complete model, not just a correctness one.
