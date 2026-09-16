# Experiment 19 — One Complete Transformer Encoder Block

**Why this experiment**: exp18 ended by naming two options — multi-head splitting, or combining exp17's residual `Add` with exp18's fused attention into one complete block. This is the latter: `y1 = LayerNorm(x + Attention(x))`, `y2 = LayerNorm(y1 + FFN(y1))` — exp2's real `TinyTransformerBlock` architecture (same post-norm residual placement), minus multi-head splitting, which stays deferred.

**The headline result**: **zero changes to `sajal_compile.py` were needed.** Every op this model uses (`Gemm`, the fused `self_attention` pattern, `Add`, `LayerNormalization`, `Gelu`) was already supported individually by exp16-18. This experiment is a test of *composition* — do independently-validated capabilities correctly combine when a real graph actually chains them together — not a test of new capability.

## A real gotcha, found before it became a false negative

Exporting the natural first version of this test model produced two `Identity` nodes at the very start of the graph, aliasing `ln1.weight`→`ln2.weight` and `ln1.bias`→`ln2.bias`. Not an architecture feature — PyTorch's exporter noticed both `LayerNorm`s started with byte-identical *default* (untrained) values (`weight=1, bias=0`) and deduplicated them. Confirmed by perturbing `ln1`/`ln2`'s weights to different random values (same as every other layer already gets) — the `Identity` nodes disappeared entirely on re-export. [python/export_full_block_test.py](../../../python/export_full_block_test.py) does this perturbation explicitly, with the reasoning stated in its own docstring, not silently.

Verified the compiler handles the *un-perturbed* version correctly too — it should, and does, fail cleanly rather than silently accept it:

```
error: op 'Identity' is not supported by this compiler outside the recognized self-attention pattern
```

## Validation

**1. Regression check on all five prior targets** — exp1's MLP, exp9's gender classifier, exp16's LayerNorm+GELU model, exp17's residual model, and exp18's single-head attention model — recompiled from scratch (compiler source unchanged, so this mainly confirms nothing about the *environment* silently drifted). **All five still bit-identical** to their respective references.

**2. Equivalence vs. PyTorch on the full block**: max abs error `4.77e-07` — larger than exp18's attention-alone number (`4.47e-08`), as expected: more chained fp32 operations (two attentions'-worth of matmuls, two LayerNorms, GELU, two residual adds) accumulate more rounding, consistent with every prior experiment's error-scales-with-depth pattern (exp2, exp11). Cosine similarity `1.0`, 100% prediction agreement.

**3. Byte-for-byte comparison against a new hand-written reference** ([native/full_block_reference.cpp](../../../native/full_block_reference.cpp)) — **bit-identical**.

**4. Negative path**: the real, un-perturbed `Identity`-containing graph rejected cleanly (above) — a genuine reachable case this time, not a contrived one.

## Interpretation

exp15 asked "can auto-generated code be correct at all" for the smallest possible case. Four experiments of incremental, individually-validated additions later, this is a real answer for something meaningfully close to a real transformer layer: **[generated_full_block.cpp](generated_full_block.cpp) is bit-identical to a hand-written implementation of the actual architecture real transformer blocks use** (post-norm residual attention + FFN), auto-generated from an ONNX export with no model-specific code written for this particular shape. The fact that zero compiler changes were needed to handle the combination is itself the finding — it's evidence that exp16-18's op-by-op validation (each checked bit-identical in isolation) actually composes, rather than each op only working in the specific narrow context it was originally tested in.

## Caveats

Still single-head — this block's attention has no per-head split, so it's not yet exp2's exact architecture (which uses 4 heads). No causal masking. The `Identity`-node gotcha is specific to untrained/default-initialized weights coinciding across layers — a real model with actually-trained weights would essentially never hit this, but a naive random-weight test-model generator (as this whole compiler-testing line of experiments uses) can, and did.

## Next experiment

Multi-head splitting remains the one piece between this and exp2's actual architecture: `Reshape`+`Transpose` to split `D` into `(num_heads, d_head)`, this experiment's fused attention computed per head, then `Transpose`+`Reshape` back to merge. Structurally a loop around the existing fused op (matching how `transformer_model.hpp`'s hand-written per-head loop already works), not new math — but the `Reshape`/`Transpose` pattern-recognition would be genuinely new parser work, unlike this experiment.
