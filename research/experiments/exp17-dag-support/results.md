# Experiment 17 — The Compiler's IR Becomes a Real DAG

**Why this experiment**: exp16 ended by naming this as the actual prerequisite for both residual connections and attention — not another op-to-function mapping like LayerNorm/GELU were, but a structural change to how the IR represents a graph at all. A flat sequential chain has no way to express "this tensor gets consumed twice, non-adjacently" — which is exactly what `y = LayerNorm(x + sublayer(x))` needs.

## What changed

[python/sajal_compile.py](../../../python/sajal_compile.py)'s IR entries now carry their *actual* resolved input tensor name(s) (from the ONNX graph directly) and their own output tensor name, instead of every op implicitly consuming "whatever the previous op produced." The parser tracks a `tensor_dims` map (ONNX tensor name → channel count), filled in as each op is processed — a `Gemm`'s weight shape retroactively tells us its input's dim too, so by the time a later `Add` node references that same tensor name again, its dim is already known.

Codegen changed to match: instead of threading one `cur` buffer through the generated function, it declares one `std::vector<float>` per unique ONNX tensor that gets produced, and each op reads its *named* input variable(s) and writes its *named* output variable. [generated_residual.cpp](generated_residual.cpp) shows this directly — line 18 uses `X` (the function parameter, aliased to the graph's own input tensor), and line 23 uses `X` *again*, non-adjacently, for the residual add. That's the actual capability this experiment adds, visible in the generated source, not just claimed.

**New op**: `Add`, restricted to exactly 2 non-initializer tensor inputs (a residual/skip connection) — bias-add is explicitly out of scope, since our `Gemm` nodes already fold bias in, and allowing `Add` with one initializer operand would silently accept a different pattern than the one this experiment is actually testing.

**New common.hpp helper**: `add_inplace()` — a two-line elementwise loop, matching the naming convention of `relu_inplace`/`gelu_inplace`.

## Test model

`y = LayerNorm(x + Linear(GELU(Linear(x))))`, then a classification head ([python/export_residual_test.py](../../../python/export_residual_test.py)) — deliberately the actual residual feed-forward half of a real transformer block (exp2/exp11-14's architecture), minus attention. Chosen specifically because `x` is consumed twice: once by the first `Linear`, again by the final `Add` — the exact pattern a flat chain can't represent. Random weights: a codegen-correctness test, not an accuracy claim.

## Validation — same bar as exp15/16, plus new checks specific to this change

**1. Regression check on all three prior targets, through the DAG-aware compiler**: exp1's MLP, exp9's gender classifier, and exp16's LayerNorm+GELU model all recompiled from scratch — all three still **bit-identical** to their respective hand-written references. The structural rewrite (flat chain → DAG) didn't silently change behavior for anything it already handled.

**2. Equivalence vs. PyTorch on the new residual model**: max abs error `4.47e-08`, cosine similarity `1.0`, 100% prediction agreement — fp32-tight, consistent with every prior experiment's numbers.

**3. Byte-for-byte comparison against a new hand-written reference** ([native/residual_reference.cpp](../../../native/residual_reference.cpp)) — **bit-identical**.

**4. Negative-path tests, three of them**:
- Re-ran exp2's transformer and exp11's BERT ONNX graphs (both contain `MatMul`/`Transpose`/`Reshape` for attention) through the DAG-aware compiler — still rejected cleanly (`op 'Identity'`/`'Shape'` not supported), confirming the wider IR didn't accidentally start accepting graphs it shouldn't.
- **A genuinely new, reachable check exercised directly**: built a model where `Add`'s second operand is a constant buffer rather than a tensor (a bias-add pattern, not a residual connection) — `error: Add with 1 non-initializer inputs, expected exactly 2`. This is exactly the kind of case exp16 flagged as *unreachable* (no supported op took two tensor inputs, so the check existed but had never actually fired) — now it's real and confirmed working.
- **The dim-mismatch check remains unverified**, honestly: constructing an `Add` between two genuinely different-dimension tensors turns out to be hard to represent as a valid ONNX graph at all — PyTorch itself refuses non-broadcastable adds at trace time, before any graph reaches this compiler. The check is still there as defense-in-depth, but — like exp16's branch-detection logic before this experiment made it reachable — it's not exercised by any test here either. Stated plainly rather than claimed as verified.

## Interpretation

This is the structural prerequisite exp16 named, done as its own isolated step rather than bundled with attention. The generated code for a genuine DAG is qualitatively different from exp15/16's chains — a tensor name gets referenced from more than one place in the generated C++, which is exactly what `git diff`-ing [generated_lngelu.cpp](../exp16-layernorm-gelu/generated_lngelu.cpp) against [generated_residual.cpp](generated_residual.cpp) shows directly: the former's `cur` variable is reassigned in a straight line; the latter's `X` parameter is read twice, non-adjacently. That's a real, visible upgrade in what the compiler can express, not just a bigger op list.

## Caveats

Still no attention. Q/K/V branching from one input is structurally similar to what this experiment solved (one tensor consumed multiple times) — but attention also needs per-head reshape/transpose and a batched matmul pattern (`MatMul` between two *runtime* tensors, not a tensor and a weight initializer, which `Gemm`'s current handling doesn't cover), a materially different and larger addition, not just "more of the same DAG machinery." No opset<20 decomposed-GELU support, unchanged from exp16. No control flow, no dynamic shapes. The dim-mismatch validation path is untested, as noted above.

## Next experiment

Attention itself, now that its main structural prerequisite (a tensor consumed more than once) is handled. The remaining new pieces, scoped honestly as a bigger step than this one: recognizing `MatMul` between two *runtime* tensors (not weight-initializer-based, unlike every `Gemm` so far), a `Transpose`/`Reshape` pair for the multi-head split (or accepting a graph that keeps heads flattened and only reshapes at the very end), and a `Div`-by-scale before the attention softmax. Given the jump in complexity, worth scoping the *smallest* attention variant first (e.g. single-head, no reshape) rather than jumping straight to exp2's full multi-head block.
