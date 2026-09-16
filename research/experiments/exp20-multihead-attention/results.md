# Experiment 20 — Multi-Head Attention

**Why this experiment**: exp18 (single-head attention) and exp19 (full encoder block) both named the same remaining gap: real transformer blocks split `D` into `(num_heads, d_head)` via `Reshape`/`Transpose`, attend per head, then merge back — exp2's actual `TinyTransformerBlock` uses 4 heads, not 1. This is the last piece between the compiler and exp2's real architecture.

**The result**: multi-head self-attention (4 heads, `d=16`, `d_head=4`) compiles to code bit-identical to a new hand-written reference, using the exact per-head BLAS-slicing trick already validated in `native/transformer_model.hpp` since exp2.

## Why this needed new parser work, unlike exp19

exp19 combined already-supported ops with zero compiler changes. This couldn't: PyTorch's ONNX exporter emits multi-head splitting as three `(Reshape, Transpose)` branches (Q, K, V) whose nodes are **not contiguous** — Q's `Reshape`/`Transpose` land right after Q's `Gemm`, but K's and V's `Gemm` nodes sit interleaved between them, e.g.:

```
Gemm(Q) → Constant×3 → Reshape(Q) → Transpose(Q) → Gemm(K) → Reshape(K) → Gemm(V) → Reshape(V) → Transpose(V) → Transpose(K) → MatMul → Mul → Softmax → MatMul → Transpose → Reshape → Gemm(O)
```

(inspected directly via `onnx.load` + printing nodes/attributes before writing any matching code — same discipline as every prior experiment). exp18's `try_match_self_attention` assumes its 5 nodes are the *next 5 nodes in the list* — that assumption breaks here. `try_match_multihead_attention` ([python/sajal_compile.py](../../../python/sajal_compile.py)) instead searches the **whole node list by tensor-name flow** (who consumes whose output), not position: find a head-splitting `Reshape`, find its `Transpose` consumer, identify K's branch by its distinct `perm=[1,2,0]` (PyTorch fuses `transpose(0,1)` and the later `transpose(-2,-1)` into one permutation for K only), trace forward through `MatMul→Mul→Softmax→MatMul→Transpose→Reshape` to the merge point, and trace each branch backward to the original Q/K/V linear-output tensor names.

A second wrinkle this created: because the pattern's nodes are scattered, the multi-head op's dependencies (K's and V's `Gemm`) aren't necessarily parsed yet when the pattern is *found* (matching starts at Q's `Reshape`, before K/V's `Gemm` nodes are reached in the sequential walk). The matcher returns the full set of consumed node indices; the caller defers appending the IR entry until the walk reaches `max(consumed indices)` — the final merge `Reshape`, which always comes after all three `Gemm`s in a validly-exported graph. This guarantees the generated C++ declares `q_weight`/`k_weight`/`v_weight`'s outputs before the fused op reads them, without needing genuine topological sorting.

## Codegen

Reuses `transformer_model.hpp`'s per-head loop unchanged in spirit: each head's `Q`/`K`/`V` is a column-slice of the full `[n, D]` buffer, addressed via BLAS's `lda`/`ldb` stride trick (pointer offset `+ h * d_head`, leading dimension stays `D`) — no physical splitting, no new numerical code. See [generated_multihead.cpp](generated_multihead.cpp)'s `for (int h = 0; h < 4; ++h)` loop.

## Validation

**1. Regression check on all six prior targets** (exp1's MLP, exp9's gender classifier, exp16's LayerNorm+GELU model, exp17's residual model, exp18's single-head attention model, exp19's full block) — recompiled from scratch with the extended parser. **All six still bit-identical** to their respective references/prior outputs. Confirms the new multi-head matcher doesn't interfere with any previously-working pattern — notably, exp18's single-head model (no `Reshape`/`Transpose` in its graph at all) still compiles through the *old* 5-node `self_attention` fusion, confirmed by grepping the regenerated code for the per-head loop (absent, as expected).

**2. Equivalence vs. PyTorch**: max abs error `8.9e-08`, cosine similarity `1.0`, 100% prediction agreement — tighter than exp18's single-head number, consistent with fewer chained ops per head (`d_head=4` vs `d=16`).

**3. Byte-for-byte comparison against a new hand-written reference** ([native/multihead_reference.cpp](../../../native/multihead_reference.cpp)) — **bit-identical**.

**4. Negative path**: corrupted one `Transpose`'s `perm` attribute in the exported graph (via direct ONNX surgery) so it no longer matches the recognized K-branch shape. The compiler rejects it cleanly: `error: op 'Reshape' is not supported by this compiler outside the recognized self-attention / multi-head-attention patterns` — exit code 1, no crash, no silent misfire.

## Interpretation

This is the piece exp18 and exp19 both deferred as "genuinely new parser work," and it turned out to be exactly that — not because the *math* changed (it's the same per-head BLAS trick exp2 already used), but because a non-contiguous, tensor-flow-based pattern match is a different kind of code than exp15-19's positional lookahead. The payoff: [generated_multihead.cpp](generated_multihead.cpp) is now compiling **exp2's actual attention shape** (4 heads), auto-generated from an ONNX export, bit-identical to hand-written code.

## Caveats

Still requires the exact export shape PyTorch's `dynamo=False` exporter produces for this `view().transpose(0,1)` idiom (K's fused `perm=[1,2,0]`, in particular) — a different PyTorch version or export path that reshapes/transposes differently would need its own pattern, not a generalization. No causal masking. `num_heads`/`d_head` must divide `D` evenly, same constraint PyTorch itself enforces at model-construction time. This experiment tests multi-head attention alone (matching exp18's scope), not yet combined with exp19's residual+FFN block — composing them is expected to work with zero further compiler changes (exp19 already proved composition holds), but hasn't been explicitly checked.

## Next experiment

Combine this with exp19's residual+FFN block to reach exp2's exact original architecture (4 heads, one full encoder block) — expected to need zero compiler changes, same as exp19 itself. Beyond that: causal masking, decomposed (opset<20) GELU, or moving from "does generated code work" toward comparing the *compiler's* cold-start/footprint numbers against this project's own hand-written baselines.
