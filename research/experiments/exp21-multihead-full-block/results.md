# Experiment 21 — Multi-Head Attention + Full Encoder Block

**Why this experiment**: exp20 named this as the next step — combine exp19's residual+FFN block with exp20's multi-head split into one graph. This is exp2's `TinyTransformerBlock` in full: 4 heads, `d_model=16`, `d_ff=64`, post-norm residual placement — the exact architecture this whole compiler line (exp15-20) has been building toward.

**The headline result**: **zero changes to `sajal_compile.py` were needed.** Same finding as exp19, one level up: exp19 already proved residual `Add` composes with attention; exp20 added multi-head splitting as an independent capability. This experiment is the test that those two compose with each other too — and they do.

## What changed vs. exp19's test model

[python/export_multihead_full_block_test.py](../../../python/export_multihead_full_block_test.py) is exp19's `TransformerEncoderBlock` with exp20's `view(seq, n_heads, d_head).transpose(0,1)` split swapped in for `self.q(x)` etc. Same Identity-node gotcha applies (default-init `LayerNorm`s export as byte-identical, deduplicated via `Identity` nodes the compiler doesn't support) — same fix as exp19: perturb `ln1`/`ln2`'s weights before export.

The resulting ONNX graph is exp19's 12-node residual+attention+FFN shape with exp20's 12-node multi-head split spliced in where exp19's single 5-node `self_attention` pattern used to be. Both of `sajal_compile.py`'s pattern matchers (`try_match_self_attention` and `try_match_multihead_attention`) coexist unmodified — the multi-head matcher claims its 12 nodes, the rest of the graph (`Add`, `LayerNormalization`, `Gelu`, the surrounding `Gemm`s) flows through exactly as it did in exp19.

See [generated_multihead_full_block.cpp](generated_multihead_full_block.cpp): Q/K/V linear ops, a 4-head loop (reusing exp20's per-head BLAS slicing), the `O` projection, residual `Add` + `LayerNorm`, the GELU-FFN, a second residual `Add` + `LayerNorm` — exactly exp2's `TransformerBlock::forward()`, auto-generated.

## Validation

**1. Regression check on all seven prior targets** (exp1's MLP, exp9's gender classifier, exp16's LayerNorm+GELU model, exp17's residual model, exp18's single-head attention model, exp19's full block, exp20's multi-head attention alone) — recompiled from scratch, compiler source unchanged. **All seven still bit-identical** to their respective references/prior outputs.

**2. Equivalence vs. PyTorch**: max abs error `4.77e-07` — matches exp19's single-head full-block number almost exactly (same order of magnitude, same op count per row), consistent with error scaling by depth of chained fp32 ops rather than by head count. Cosine similarity `1.0`, 100% prediction agreement.

**3. Byte-for-byte comparison against a new hand-written reference** ([native/multihead_full_block_reference.cpp](../../../native/multihead_full_block_reference.cpp), exp2's exact `TransformerBlock` shape) — **bit-identical**.

## Interpretation

This closes the loop this compiler line opened at exp15: **[generated_multihead_full_block.cpp](generated_multihead_full_block.cpp) is bit-identical to a hand-written implementation of exp2's actual architecture**, auto-generated from an ONNX export, with a compiler whose every individual capability (Linear, ReLU/GELU/Softmax, LayerNorm, residual Add, single-head attention, multi-head splitting) was validated bit-identical in isolation first, and which now composes all of them into a real transformer block with zero glue code written specifically for this combination. The two "zero compiler changes needed" results (exp19, exp21) aren't luck — DAG-based IR (exp17) and pattern-fusion into named-tensor IR ops (exp18, exp20) were deliberately designed so that composition doesn't require new code, and both times it didn't.

## Caveats

Same as exp19 and exp20 individually: no causal masking, no decomposed (opset<20) GELU support, still needs the exact `Reshape`/`Transpose` shapes PyTorch's `dynamo=False` exporter emits for this specific `view().transpose()` idiom. This is one block — a real model (e.g. exp2's own multi-layer stack, or BERT) chains several; nothing here tests multi-block graphs, though the DAG-based IR has no obvious reason to require new work there either.

## Next experiment

The compiler now handles a complete, real single-block transformer architecture. Candidates from here: stack multiple blocks (does the IR/codegen handle N repeated blocks fed into each other, still zero new code?); benchmark the *compiled* artifact's cold-start/footprint numbers against this project's own hand-written exp2 baseline, closing the loop on the project's original value proposition rather than only its correctness; or decomposed-GELU support to widen which ONNX opsets/exporters this compiler accepts.
