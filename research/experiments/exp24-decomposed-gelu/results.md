# Experiment 24 — Decomposed GELU (opset<20)

**Why this experiment**: every prior compiler experiment exported at `opset_version=20`, where ONNX GELU is a single fused `Gelu` node. `Gelu` only became a fused ONNX op at opset 20 — any model exported at an older opset (which is most real-world exports, since opset 20 only shipped in late 2023) decomposes GELU into 5 primitive ops instead. exp22 and exp23 both found real gaps by testing against something closer to a real model than a synthetic one; this widens what counts as "a real model" the compiler accepts.

**The result**: recognized cleanly, no surprises this time — inspecting the graph first (as always) showed the decomposed pattern is a strictly sequential chain, not a branching one, so a simple positional lookahead (exp18's style) was the right tool, not exp23's tensor-flow tracing.

## The exact pattern

Exported a plain `Linear -> GELU -> Linear` model at `opset_version=17` and read the graph before writing any matching code:

```
Div(x, 1.41421356)  ->  Erf  ->  Add(·, 1.0)  ->  Mul(x, ·)  ->  Mul(·, 0.5)
```

This is `0.5 * x * (1 + erf(x / sqrt(2)))` — the exact GELU formula, decomposed into ONNX primitives that predate opset 20. `x` (the original tensor, before the `Div`) is reused twice: once as `Div`'s numerator, once as the third node's other `Mul` operand — the same tensor referenced from two places in the chain, not a fresh copy.

## Why this pattern doesn't need exp23's tensor-flow tracing

Multi-head attention's `Reshape`/`Transpose` branches interleave with other nodes (Q/K/V's separate `Gemm` calls sit between them), which is why exp23 had to rewrite that matcher to trace specific tensor names rather than assume positional contiguity. Decomposed GELU has no such structure — it's one linear chain, each node consuming only the previous node's output (plus one reused reference back to `x`). `try_match_decomposed_gelu` ([python/sajal_compile.py](../../../python/sajal_compile.py)) is a straightforward 5-node lookahead, same shape as `try_match_self_attention`, checking each op type and validating the three constants (`√2`, `1.0`, `0.5`) against their expected values with `math.isclose` (opset-exported constants carry float32 rounding, so exact equality would be too strict).

The matched IR entry is just `{"op": "gelu", ...}` — identical to the fused-`Gelu` case exp16 already handles. No codegen changes at all; only the parser gained a second way to recognize the same op.

## Validation

**Regression, all 10 prior compiler targets** (exp1, exp9, exp16-23) recompiled from scratch — still bit-identical.

**Isolated case**: a `Linear->GELU->Linear` model at opset 17 compiles and matches PyTorch (max abs error `1.19e-07`).

**The real integration test**: exp21's full multi-head attention + residual + FFN block ([python/export_multihead_full_block_test.py](../../../python/export_multihead_full_block_test.py)'s model), re-exported at **opset 17 instead of 20** — same architecture, only the export opset differs. Inspected the resulting graph first: attention's `Reshape`/`Transpose`/`MatMul` structure and `LayerNormalization` are unaffected by the opset change (both have existed since well before opset 17); only GELU decomposes. Compiles cleanly with zero additional changes beyond the GELU matcher itself. PyTorch equivalence: max abs error `4.77e-07` — matching exp21's opset-20 number exactly (same architecture, same weights convention, just a different export path to the same IR). **Bit-identical** against exp21's existing hand-written reference ([multihead_full_block_reference.cpp](../../../native/multihead_full_block_reference.cpp)) — the same reference file, no changes, since the architecture didn't change.

**Negative path**: corrupted the `0.5` constant to `0.6` in the exported graph. Rejected cleanly: `error: op 'Div' is not supported...` — exit 1, no silent misfire.

## Interpretation

This is the narrowest-scope experiment in the compiler line so far: one matcher function, no codegen changes, no IR changes, validated against both an isolated case and a full real-architecture graph. The interesting confirmation is the integration test — proof that "decomposed GELU" and "multi-head attention + residual + LayerNorm" are independent capabilities that compose without any interaction, the same kind of result exp19/21 found for other capability pairs (though here, unlike those, no new bug turned up — the two patterns genuinely don't touch each other's nodes).

This also means the compiler now accepts models exported at ANY opset from 17 through 20 for every architecture tested so far, not just the one opset every prior experiment happened to use.

## Caveats

Only the exact 5-node decomposition PyTorch's exporter emits — a different exporter (or a future PyTorch version) producing e.g. `Tanh`-based GELU approximation, or the same math with operands in a different but equivalent order the current node-by-node checks don't anticipate, would still fail cleanly rather than being silently mishandled. `Erf`-based (exact) GELU only — the `tanh` approximation formula (`approximate='tanh'` in PyTorch) is a different, unrelated decomposition, not attempted.

## Next experiment

The compiler has now accumulated: Linear, ReLU/GELU(fused or decomposed)/Softmax, LayerNorm, residual Add, single- and multi-head attention (`Mul`- or `Div`-scaled), and multi-block stacking. Candidates from here: causal attention masking (needed for autoregressive models, not yet attempted); or moving toward a small end-to-end demonstration — compiling and benchmarking a real, publicly-downloaded small transformer (not one of this project's own hand-rolled test models) to see whether all these individually-validated capabilities hold up on a graph nobody in this project wrote.
