# Experiment 23 — Stacking Multiple Encoder Blocks

**Why this experiment**: exp21/22 both named this as the next open question — does the compiler handle N repeated transformer blocks feeding into each other, or does it only work for the single-block graphs every prior experiment (exp15-22) happened to test?

**The result, in two parts**: it doesn't work for free — a real bug surfaced (exp20's original multi-head matcher breaks on more than one attention instance in the same graph) — but the fix is small, and once made, N-block stacks compile correctly with no further changes.

## Inspecting the graph first, as always

Before assuming anything, exported a 2-block stack (`nn.ModuleList` of exp21's `MultiHeadTransformerEncoderBlock`) and read the ONNX graph directly. PyTorch scopes each block's tensor names distinctly — `blocks.0.q.weight` vs. `blocks.1.q.weight`, `/blocks.0/Reshape_output_0` vs. `/blocks.1/Reshape_output_0` — no naming collision, and block 1's input is literally block 0's output tensor, chained the way you'd expect. This looked like it should just work.

It didn't. First compile attempt: `error: op 'Reshape' is not supported...`.

## The actual bug: a global search that doesn't scope to one attention instance

exp20's `try_match_multihead_attention` found its three `Reshape`/`Transpose` branches by **scanning the whole node list for any Reshape whose target shape matches** `(num_heads, d_head)`. With one attention block in the graph, that's exactly 3 matches (Q/K/V) — correct by coincidence, since there was only ever one instance to find. With two IDENTICALLY-SHAPED blocks stacked, that same scan finds *6* matching reshapes (3 per block), fails the `len(reshape_group) != 3` check, and the whole match silently returns `None` — every block, not just the second, since the very first attempted match (at block 0's Q-reshape) already sees all 6.

This is exactly the class of bug a toy single-instance test model structurally cannot expose — every prior compiler experiment (exp18, exp20, exp21, exp22) had exactly one attention pattern per graph, so "search the whole graph for anything shaped like this" and "search for the specific instance connected to this branch" gave identical results. Stacking two blocks is the first graph where they diverge.

## The fix: purely tensor-flow-traced, never a global shape scan

Rewrote `try_match_multihead_attention` ([python/sajal_compile.py](../../../python/sajal_compile.py)) so every lookup traces a *specific* tensor name forward (who consumes this exact output) or backward (who produces this exact input) from wherever the match started — never "find anything shaped like X anywhere in the graph." Concretely: from the triggering `Reshape`+`Transpose` branch, find the one `MatMul` that consumes its specific output tensor; look at what produces that `MatMul`'s *other* input to tell whether the branch is Q/K (other input comes from another `Transpose`) or V (other input comes from a `Softmax`); from there, trace the rest of the pattern (scale, softmax, the second `MatMul`, the merge) the same way. Every step is anchored to one real tensor name, so it can never widen to a different attention instance elsewhere in the graph, no matter how many other identically-shaped blocks exist.

This is a genuinely different algorithm from exp20's, not a patch — but the *math* and the codegen it feeds are completely unchanged.

## Validation

**Regression, all 9 prior compiler targets** (exp1, exp9, exp16-22) recompiled from scratch after the rewrite — still bit-identical. This matters more than usual here: a match-scoping rewrite is exactly the kind of change that could silently break the single-instance case while fixing the multi-instance one.

**2-block stack** ([python/export_stacked_blocks_test.py](../../../python/export_stacked_blocks_test.py), exp21's block shape × 2): compiles cleanly; the generated code contains two independent 4-iteration per-head loops (confirmed by inspection, [generated_stacked_blocks.cpp](generated_stacked_blocks.cpp)), one per block, each correctly wired to that block's own Q/K/V. PyTorch equivalence: max abs error `4.77e-07`. Bit-identical against a new hand-written reference ([native/stacked_blocks_reference.cpp](../../../native/stacked_blocks_reference.cpp)) — literally exp21's reference struct, called twice.

**3-block stack** (ad hoc, not committed as a separate artifact): compiles cleanly, cosine similarity `~1.0` vs. PyTorch — confirms the fix generalizes to N, not just the specific N=2 case that exposed the bug.

## Interpretation

The honest framing: this experiment did not confirm "composition works for free" the way exp19 and exp21 did — it found a real correctness bug the first time a genuinely new graph shape (more than one of the same pattern) was tried, which is exactly why the project's discipline is "inspect the real graph, don't assume" rather than "the design should generalize, so it does." The bug was subtle specifically because every existing regression test was structurally incapable of catching it (each has exactly one attention instance) — a reminder that a green regression suite only proves "no regression among cases already covered," not "no bugs."

Once fixed, though, the result is what exp21 predicted: a real (small) transformer stack compiles with zero *additional* op-level capability, using the same DAG-based IR (exp17) and per-head BLAS codegen (exp20) validated all along.

## Caveats

Still toy scale (`d=16`, `seq_len=8`) for the committed regression artifact — not re-benchmarked at exp2's real scale (exp22 already established the compiler carries no per-block overhead at 1 block; N blocks is N× the same per-head loop, with no reason to expect a different result, but this wasn't re-measured). All blocks share the same shape in this test; a real model stacking blocks of different sizes wasn't tried.

## Next experiment

Widen the accepted op set (decomposed opset<20 GELU, causal masking) to accept more real-world exported models as-is, since exp22 already showed testing against a real model (not just synthetic ones) is what actually finds gaps like the `Div`-vs-`Mul` one and this experiment's global-scan bug.
