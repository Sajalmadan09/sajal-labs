# Experiment 30 — GPT-2's Additive Causal Mask

**Why this experiment**: exp29 closed the combined-QKV `Split` gap and confirmed, by actually re-testing, that real GPT-2's fully-unrewired trace still failed — on a different, separate gap: GPT-2's own causal mask decomposes as `Equal`+`Where`+`Add` (an additive bias combined separately), not exp27's `Cast`+`Where` (a direct select). This closes that gap, and with it, the last piece named since exp25: **real GPT-2, traced directly by Hugging Face's own code, now compiles with zero rewiring at all.**

## Getting the polarity right required checking, not deriving

Inspecting the graph (as always) showed the shape: `Trilu(Expand(1.0, dynamic_shape)) -> Equal(·, 0.0) -> Where(mask, -inf, 0.0) -> Add(scores, ·)`. Four real differences from exp27's synthetic pattern at once: the all-ones matrix comes from a runtime-shaped `Expand`, not a literal `Constant`; the mask is inverted through `Equal` rather than passed through `Cast`; the combine is additive (`Add`) rather than a direct `Where`-on-scores; and — the one that actually caused a wrong first attempt — `Trilu`'s `upper` attribute.

A first hand-derivation, assuming `upper` defaulted to the ONNX spec's value of 1 (since the attribute hadn't been printed in the initial node dump), concluded the pattern masked *past* positions and left *future* ones visible — backwards, and worth doubting immediately given GPT-2 obviously works. Rather than keep re-deriving by hand, two things settled it directly: (1) an empirical test — replace the mask with the standard "mask column > row" condition, run the rest of one real block's math by hand in PyTorch, and compare against calling `model.h[0]` directly; the standard condition matched to `5.7e-06`, confirming what the *net* mask has to do regardless of how the ONNX graph spells it. (2) Extracting the actual `Trilu` node's runtime output via `onnxruntime` (adding it as a graph output and running the model) showed `upper=0` explicitly set — not the default. With that one fact, the by-hand derivation reduces cleanly to the standard condition. Two independent checks, not one assumption stacked on another.

## The generalized matcher

`_causal_bool_condition` ([python/sajal_compile.py](../../../python/sajal_compile.py)) now reads `Trilu`'s actual `upper` attribute and `k` (its 2nd input if present, default 0) rather than assuming either, and traces the boolify step as either `Cast` (pass-through) or `Equal(x, 0)` (inverted) — computing the *net* row/column condition symbolically and requiring it reduce to exactly "column > row" (the one causal shape this compiler claims to support), regardless of which concrete `(upper, k, inverted)` combination a given exporter happens to produce. `try_match_causal_mask` and `unwrap_causal_mask_backward` each now try two combine shapes: exp27's direct `Where(mask,-inf,scores)`, and GPT-2's additive `Where(mask,-inf,0)` + `Add(scores,·)`.

## Two more real bugs, found only by trying to compile the real trace

Even with the mask logic right, compiling failed twice more, each caught by reading exactly where and how:

1. **K's Reshape has two Transpose consumers, not one.** GPT-2's real export reuses K's head-split reshape for two purposes: the actual `K^T` used in the attention matmul (`perm=[0,2,3,1]`), and a second "throwaway" transpose (`perm=[0,2,1,3]`) that feeds *only* the mask's dynamic-shape computation (`Shape→Slice→Concat`), never the real math. `branch_from_reshape` had always assumed exactly one consumer — true for every model tried through exp29. Fixed by trying each valid-looking candidate and preferring whichever one is actually consumed (directly or through a scale-wrap) by a `MatMul`; the throwaway one never is.
2. **The orphaned dead-end subgraph needed explicit marking, not silent tolerance.** The unused transpose's own `Shape`/`Slice`/`Concat` chain (computing the `Expand`'s shape input) isn't consumed by anything else either — it's genuinely dead, since this compiler already knows the sequence length at compile time (`n` is a runtime parameter, not something to recompute from tensor shapes). `_consume_shape_bookkeeping_chain` walks backward through only `Shape`/`Slice`/`Concat` nodes, stopping the instant it hits anything else (so a *real*, still-needed tensor whose shape happens to feed this chain — like Q's actual transpose — is correctly left alone), marking the dead nodes consumed so the main walk doesn't later reject them as unrecognized standalone ops.

## Validation

**Regression, all 19 prior compiler targets** (exp1, exp9, exp16-29) recompiled from scratch after every change — still bit-identical, including exp27's own two causal-mask targets (the original `Cast`+direct-`Where` shape, now one of two recognized combine styles instead of the only one).

**Real GPT-2, traced directly, zero rewiring** ([python/export_real_gpt2_direct_test.py](../../../python/export_real_gpt2_direct_test.py) — `model.h[:2](x)`, no wrapper rewiring at all, unlike exp28's `export_real_gpt2_test.py`): compiles. PyTorch equivalence vs. calling the real layers directly: max abs error `1.22e-04` — matching exp28's rewired-wrapper number to within statistical noise, at the same real scale (hidden=768).

**Cross-check between the two paths**: compared this experiment's direct-compile output against exp28's rewired-compile output (same real weights, same architecture, two structurally different ONNX graphs). Not bit-identical (`1.14e-05` max diff) — expected and consistent with exp26's own finding: the rewired path uses a single post-matmul scale, the direct path uses GPT-2's real split pre-matmul scaling (`d_head**-0.25` applied to Q and K separately); combining two independently-rounded factors doesn't exactly recover computing the scale once. Same understood, benign floating-point non-associativity, not a new bug.

**Negative path**: flipped `Trilu`'s `upper` attribute from `0` to `1` (inverting the mask's actual meaning). Rejected cleanly — the derived condition no longer reduces to the standard causal shape, so the whole attention match fails and the graph is rejected at `Split` (the earliest node whose fate depended on that match succeeding), not silently computed with a wrong mask.

## Interpretation

This closes the loop opened at exp25: a real Hugging Face model, traced with zero modifications, zero Python-side weight rewiring, compiling and matching its own real output. Getting here from exp28's "named, deferred" starting point took three independent, unanticipated obstacles (the mask's exact polarity, a duplicate Transpose consumer, an orphaned dead subgraph) — each found only by attempting the actual real trace and reading precisely how it failed, not by extending the design in the abstract. The pattern holds across the whole GPT-2 arc (exp28→29→30): synthetic test models validate a mechanism's *shape*, but real exports are what reveal the mechanism's edge cases.

## Caveats

Same scope as exp28/29: the decoder-block stack alone (no embeddings, no `lm_head`), and only the specific `(upper, k, inverted, combine-style)` combinations actually observed are recognized — a mask expressed some third way (e.g. via `ConstantOfShape` instead of `Expand`, or a sliding-window rather than fully-causal pattern) would still fail cleanly rather than being guessed at.

## Next experiment

The GPT-2 arc (exp28-30) is functionally complete for a 2-block real decoder. Candidates from here: stack more of GPT-2's real layers (does depth alone surface anything new, the way exp23 found for synthetic stacks?); or move to a different architecture family entirely (encoder-decoder, e.g. T5) to see what class of new gap that surfaces, continuing the same discipline — real models find what synthetic ones can't.
