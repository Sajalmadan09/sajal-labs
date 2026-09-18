# Experiment 27 — Causal Attention Masking

**Why this experiment**: every attention model this compiler has accepted through exp26 was bidirectional (encoder-only, every token attends to every other token) — real bert-tiny included. Autoregressive/decoder models (GPT-style) need each position to attend only to itself and earlier positions. exp26 named this as the next step.

**The result**: recognized cleanly on the first real attempt, no surprises — inspecting the exported graph first showed a single, simple pattern, and both attention matchers (exp18's positional one, exp20/26's tensor-flow-traced one) needed the same small addition.

## The exact pattern

Exported a `scores.masked_fill(causal_mask, -inf)` model (PyTorch's standard idiom: `causal_mask = torch.triu(torch.ones(seq,seq,dtype=bool), diagonal=1)`) and read the graph before writing any matching code:

```
Trilu(ones_matrix, k=1) -> Cast -> Where(mask, -inf, scores)
```

sitting between the scale step and `Softmax`. `Trilu` with `k=1` computes the strict upper triangle of the all-ones matrix (excluding the diagonal); `Cast` converts it to bool; `Where` picks `-inf` where the mask is true (future positions) and the original score otherwise. This is the ONNX-primitive decomposition of the masking call — no fused "causal mask" op exists at this opset, similar in spirit to exp24's decomposed GELU.

## Where it plugs into the existing matchers

Both attention pattern-matchers needed the same conceptual addition — "optionally see past a mask between the scale and softmax steps" — via a shared validator, `_validate_causal_mask` ([python/sajal_compile.py](../../../python/sajal_compile.py)), checking the exact shape (`Trilu`'s input is an all-ones square matrix, `k=1`) and fill value (`-inf`, not just a large negative number). Two thin wrappers use it in the two directions each matcher needs:

- `try_match_causal_mask(nodes, constants, scores_tensor)` — **forward**: "does scores_tensor feed a mask before reaching softmax?" Used by `try_match_self_attention` (positional lookahead — exp18's style) and by `try_match_multihead_attention`'s Q/K-triggered branch (which already had the pre-softmax tensor in hand from `unwrap_scale_forward`).
- `unwrap_causal_mask_backward(nodes, constants, tensor_name)` — **backward**: "is tensor_name itself a mask's output — what did it mask?" Used by `try_match_multihead_attention`'s V-triggered branch, which reaches softmax's input by tracing *backward* from softmax and needs to see past an optional mask to find the pre-softmax `MatMul`.

Both attention IR entries (`self_attention`, `multihead_self_attention`) gained a `causal` flag. Codegen ([common.hpp](../../../native/common.hpp)) gained one new primitive, `causal_mask_rows(scores, n)` — a plain double loop setting `scores[i][j] = -INFINITY` for `j > i` — called once per head, right before `softmax_rows()` (which already handles `-inf` entries correctly: they `exp()` to `0` and drop out of the row sum, so no other code needed changing).

## Validation

**Regression, all 14 prior compiler targets** (exp1, exp9, exp16-26) recompiled from scratch — still bit-identical. This matters here specifically: confirms the new optional mask-detection branch in both matchers doesn't accidentally fire (or change behavior) on any of the many *non*-causal attention graphs already covered.

**Causal multi-head attention** ([python/export_causal_attention_test.py](../../../python/export_causal_attention_test.py), 4 heads, d=16): PyTorch equivalence max abs error `5.96e-08`. **Bit-identical** against a new hand-written reference ([native/causal_attention_reference.cpp](../../../native/causal_attention_reference.cpp)).

**Causal single-head attention** ([python/export_causal_single_head_test.py](../../../python/export_causal_single_head_test.py)) — the *other* matcher, exercised independently since it detects the mask via a different code path (positional vs. tensor-flow-traced): PyTorch equivalence max abs error `5.96e-08`. **Bit-identical** against [native/causal_single_head_reference.cpp](../../../native/causal_single_head_reference.cpp).

**Negative path**: corrupted the mask's diagonal offset (`Trilu`'s `k` from `1` to `0`, meaning "attend to yourself and earlier, plus nothing extra" becomes a *different*, non-standard mask this compiler doesn't claim to recognize). The compiler doesn't silently accept it as ordinary (non-causal) attention — the malformed mask still sits between the scale step and softmax, so `find_consumers` looking for softmax's direct consumer finds the `Where` node instead and fails; the whole attention match returns `None` and the graph is rejected cleanly (`error: op 'Reshape' is not supported...`). This is the important property: an unrecognized masking scheme fails loud, it never gets silently computed as if unmasked (which would produce confidently wrong output).

## Interpretation

Unlike exp23/25/26, which each found the "obvious" approach broken by something a real graph does differently, this one worked close to how it was designed on the first real attempt. The one design choice worth naming: causal masking is recognized as a *modifier* on the existing attention IR ops (`causal: bool`), not a new IR op of its own — the actual computation is unchanged except for one extra loop before softmax, so there was no reason to model it as a separate capability. This mirrors exp24's decomposed-GELU precedent (a different ONNX shape for the same op, not a different op) more than exp20's original multi-head split (which genuinely needed new codegen).

## Caveats

Only the standard strict upper-triangular mask (`Trilu(ones, k=1)`, exact `-inf` fill) is recognized — a sliding-window mask, a padding mask combined with causal masking, or a mask expressed via `Slice`+`ConstantOfShape` instead of `Trilu` would fail cleanly rather than being handled. Not tested against a real decoder model (e.g. GPT-2) — only synthetic single- and multi-head test models, matching exp18/20's original scope before those were later validated against exp2 and real bert-tiny.

## Next experiment

Try this against a real small decoder model (e.g. GPT-2's smallest checkpoint) the way exp25/26 did for BERT — that's historically where this project's synthetic-test-model assumptions have actually broken, not where they've held.
