# Experiment 18 — Single-Head Attention in the Compiler

**Why this experiment**: exp17 named attention as the next step but flagged it as a *bigger* jump than the DAG rewrite — needing `MatMul` between two runtime tensors (not weight-based, unlike every `Gemm` so far) and per-head reshape/transpose. This scopes the smallest real slice of that: single-head attention, no reshape needed at all since there's only one head — isolating the genuinely new capability (runtime-tensor `MatMul`) from multi-head splitting, which stays deferred.

## The real complication, found by inspecting the actual ONNX graph first

Single-head attention (`Q,K,V = Linear(x)`, `scores = Q@K^T * scale`, `attn = softmax(scores)`, `ctx = attn@V`, then an output projection) exports to five new-to-this-compiler nodes: `Transpose(K, perm=[1,0])`, `MatMul(Q, K^T)`, `Mul(scores, scale_constant)`, `Softmax`, `MatMul(attn, V)`. The awkward part: the `[seq_len, seq_len]` score matrix's width is `seq_len` — a value only known at runtime (`n`, the row count passed into `forward()`), not a compile-time constant the way every previous op's dimensions were (always derived from a fixed weight shape). Generalizing the whole dimension-tracking system to support symbolic/runtime dimensions everywhere would have been a large, invasive change.

**The fix: don't generalize dimension tracking — fuse the whole pattern into one op instead.** [python/sajal_compile.py](../../../python/sajal_compile.py) recognizes this exact 5-node shape (`Transpose→MatMul→Mul→Softmax→MatMul`) as a single `self_attention` IR entry. Because the fusion happens at recognition time, the `[n,n]` score matrix never becomes a separate IR tensor that the rest of the system would need to reason about — it's purely an internal detail of the fused op's own codegen. Every dimension the *surrounding* graph sees is still a normal fixed constant (the feature dim `D`). This is a materially narrower claim than "supports MatMul and Transpose" — stated as such, not implied to be more general.

**Codegen reuses validated math, not new numerical code**: the fused op emits exactly the same two-`cblas_sgemm`-call pattern already proven in [native/transformer_model.hpp](../../../native/transformer_model.hpp) and [native/bert_model.hpp](../../../native/bert_model.hpp) — `CblasTrans` on K computes `Q@K^T` with no physical transpose, matching the BLAS trick those hand-written headers already used. exp18 doesn't invent attention math; it makes already-validated attention math reachable from a compiled graph instead of only from hand-written code.

## Test model

Single-head self-attention, `d=16`, `seq_len=8` ([python/export_attention_test.py](../../../python/export_attention_test.py)). One real difference from exp15-17's test harness, worth being explicit about: `test_config.txt`'s `n_test` now doubles as *sequence length* — attention mixes information across all rows in one `forward()` call (they attend to each other), unlike every previous op which treated rows independently. This is one sequence, not N independent examples, matching how exp2/exp11-14's real attention was always tested.

## Validation — same bar as exp15-17

**1. Regression check on all four prior targets**: exp1's MLP, exp9's gender classifier, exp16's LayerNorm+GELU model, and exp17's residual model — all four recompiled from scratch through the attention-aware compiler, **all four still bit-identical** to their respective references. Adding attention-pattern recognition didn't touch anything about how the existing ops compile.

**2. Equivalence vs. PyTorch**: max abs error `4.47e-08`, cosine similarity `1.0`, 100% prediction agreement — the correctly-extracted scale constant (`0.25`, i.e. `16^-0.5`, read directly from the ONNX `Constant` node rather than hand-typed) is visible in [generated_attention.cpp](generated_attention.cpp)'s `cblas_sgemm` call.

**3. Byte-for-byte comparison against a new hand-written reference** ([native/attention_reference.cpp](../../../native/attention_reference.cpp)) — **bit-identical**.

**4. Three negative-path tests**:
- exp2's real (multi-head) transformer and exp11's real BERT — still rejected cleanly (`Identity`/`Shape` not supported), confirming the new pattern-matcher doesn't accidentally widen what's accepted.
- **A deliberately malformed near-attention graph** — the same Q/K/V/softmax structure but with the scale-`Mul` step removed (unscaled attention) — correctly does **not** match the 5-node pattern (softmax appears one position early) and falls through to standalone-op rejection (`Transpose` not supported), rather than silently matching something it shouldn't or crashing. This is the check exp16/exp17 couldn't exercise (no reachable false-positive case existed yet); this experiment has one and it behaves correctly.

## Interpretation

This validates that fusion-by-pattern-recognition — deferred as "too complex" for GELU's decomposed form back in exp16 — is a viable, honest strategy when the alternative (generalizing dimension tracking to support runtime-symbolic shapes) would be considerably more invasive for a single use case. The explicit trade-off: this compiler now handles *exactly* this one attention shape, not attention in general. A model with even a slightly different but mathematically-equivalent export (e.g. `Div` instead of `Mul` for the scale, or the scale applied to Q before the matmul instead of to the scores after) would currently be rejected, correctly, rather than silently mishandled — narrow-but-honest over broad-but-fragile, consistent with every prior experiment's scoping choices.

## Caveats

Single head only — no per-head reshape/transpose, still explicitly deferred. Exactly one scale-application order (`Mul` after `MatMul`, applied to the full scores) and exactly one attribute shape (`Transpose` with `perm=[1,0]`, `Mul` by a `Constant`) are recognized; mathematically-equivalent variations are not, by design (see Interpretation). No causal/attention masking (this test model has none — a real decoder would need it, a genuinely separate addition). No opset<20 decomposed-GELU, unchanged from exp16.

## Next experiment

Multi-head attention itself is now the main remaining piece for a real transformer block: `Reshape`+`Transpose` to split `D` into `(num_heads, d_head)`, the same fused-attention computation *per head*, then `Transpose`+`Reshape` back to merge — structurally a loop around this experiment's fused op, not a new mathematical capability. Alternatively, combining exp17's residual `Add` with this experiment's attention (attention wrapped in `LayerNorm(x + Attention(x))`) would complete one full, real, single-head transformer *encoder* block, closer to exp2's architecture than either experiment alone — likely the more natural next step before tackling multi-head splitting.
