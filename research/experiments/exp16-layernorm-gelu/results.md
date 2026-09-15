# Experiment 16 — LayerNorm and GELU in the Compiler

**Why this experiment**: exp15's own "next experiment" section named this as the immediate next step — widen op coverage incrementally, LayerNorm and GELU before attention, same bit-identical-vs-hand-written validation bar. This does exactly that, and along the way replaces exp15's hardcoded 4-op template with genuinely generic IR-driven codegen.

## What changed in the compiler

[python/sajal_compile.py](../../../python/sajal_compile.py) now recognizes 5 ONNX op types (`Gemm`, `Relu`, `LayerNormalization`, `Gelu`, `Softmax`, up from 3) and — more importantly — the codegen is no longer a fixed template filled with four blanks. It walks the IR list and emits one block of C++ per op, so the generated code's length and structure now genuinely vary per model rather than always being the same four lines with different numbers substituted in. [generated_lngelu.cpp](generated_lngelu.cpp) is the actual output for this experiment's test model — worth comparing against exp15's [generated_exp1_mlp.cpp](../exp15-minimal-compiler/generated_exp1_mlp.cpp) to see the codegen genuinely responding to a different graph shape.

**A real ONNX quirk, checked before writing any parsing code, not assumed**: PyTorch's exporter emits `LayerNormalization` as a single fused op at opset 17+, but decomposes `GELU` into five primitive ops (`Div → Erf → Add → Mul → Mul`) unless the export targets opset 20+, where it becomes one fused `Gelu` node. This experiment's test model is exported at opset 20 specifically to get the fused op — pattern-matching the decomposed 5-node form is explicitly deferred, not silently unsupported-but-untested.

**A real correctness trap, caught before it shipped**: exp15's compiled output reused `shapes.txt`'s 4-field convention because it happened to coincide with `mlp_model.hpp`'s fixed shape. Once LayerNorm/GELU are in the mix, that coincidence no longer holds — reusing the same convention would let `sajal`'s existing "mlp kind" detection silently misinterpret a richer graph as a plain MLP and run the *wrong* forward pass. Switched to a new `test_config.txt` convention (just `in_dim n_test`) specifically to avoid this; `sajal` doesn't recognize compiler output from this experiment yet, stated as a real caveat below, not swept under the rug.

## Test model

`Linear(32→64) → LayerNorm → GELU → Linear(64→10) → Softmax` ([python/export_lngelu_test.py](../../../python/export_lngelu_test.py)) — a real, if synthetic, architecture chosen specifically to exercise both new ops without also requiring attention's residual/branching structure (see Scope below). Random weights: this experiment is about codegen correctness, not an accuracy claim, matching exp5/exp12's use of synthetic weights for mechanism-isolation.

## Validation — same four checks as exp15

**1. Regression check on exp15's original two targets, through the now-generalized compiler**: recompiled exp1's MLP and exp9's gender classifier from scratch — both still bit-identical to their hand-written references. The generalization didn't silently change behavior for the cases it already handled.

**2. Equivalence via a new `compare_compiled.py`** (same methodology as `compare.py`, adapted for the `test_config.txt` convention — see below for why it isn't the same script unchanged this time):

| | Max abs error vs. PyTorch | Cosine similarity | Prediction agreement |
|---|---:|---:|---:|
| LayerNorm+GELU test model | 2.98e-08 | 1.0 | 100% |

**3. Byte-for-byte comparison against a new hand-written reference** ([native/lngelu_reference.cpp](../../../native/lngelu_reference.cpp), written independently of the compiler, same architecture) — **bit-identical**, matching exp15's validation bar exactly.

**4. Negative-path test — does it still fail cleanly on graphs outside its scope?** Built a genuine residual-connection model (`LayerNorm(x + Linear(x))`, the pattern every real transformer block uses) and compiled it:

```
error: op 'Add' is not supported by this compiler (supported: ['Gelu', 'Gemm', 'LayerNormalization', 'Relu', 'Softmax'])
```

Rejected cleanly — worth being precise about *how*, though: this fired at the "unsupported op type" check (`Add` was never added to the supported set), not at the separate "non-sequential inputs" branch-detection logic the parser also contains. Since every currently-supported op takes exactly one non-initializer input by construction, that branch-detection code is currently unreachable/untested — real defense-in-depth for when `Add` is eventually added (residual connections need it), but not yet exercised by any test. Stated honestly rather than claimed as verified.

## What this experiment establishes vs. doesn't

**Establishes**: the compiler generalizes to a wider, genuinely IR-driven op set without regressing exp15's original targets; LayerNorm and GELU recognition + codegen is correct to the same bit-identical bar as exp15; the "detect unsupported ops and fail loudly" property extends naturally to new op types without needing new code, since it's a property of the `SUPPORTED_OPS` set check, not something special-cased per op.

**Does not establish**: that the branch/residual-detection logic actually works (untested, as noted above — the real test needs `Add` to be a supported op first, which needs the IR to become a DAG, which is exactly the harder step attention also requires); that decomposed (opset<20) GELU subgraphs are handled (explicitly out of scope, real-world ONNX exports at common opsets will hit this); anything about attention or `Embedding`, still fully deferred to a later experiment as planned.

## Caveats

Sequential-chain-only compiler, still. Real transformer blocks need residual connections, which need the IR to represent a DAG (multiple named tensors, ops that reference non-adjacent prior outputs) — a structural change to the compiler, not just a new op-to-function mapping like this experiment's additions were. `sajal` CLI doesn't recognize this experiment's `test_config.txt` convention — extending it is possible but wasn't needed to answer this experiment's question and would have been scope creep.

## Next experiment

Two independent next steps, still deliberately not bundled: (1) support the opset<20 decomposed-GELU subgraph pattern, since that's what real-world exported models will actually contain far more often than opset 20's fused form; (2) the harder structural step — generalize the IR from a flat sequential list to a proper DAG (each op references named input tensors, not just "the previous op's output"), which is the actual prerequisite for both residual connections and attention's Q/K/V branching, and would finally let the "detects non-sequential inputs" safety logic be genuinely tested rather than just present in the code.
