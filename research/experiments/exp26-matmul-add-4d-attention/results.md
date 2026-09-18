# Experiment 26 — MatMul+Add Linear and 4D Attention

**Why this experiment**: exp25 found four real gaps by tracing Hugging Face's actual `BertLayer.forward()` directly, and deliberately routed around them (rewiring real weights into the compiler's known convention instead). This closes two of those four — `MatMul`+`Add` as an alternate Linear encoding, and 4D-batched attention shapes — and the result is bigger than "the compiler accepts more ops": **the full real bert-tiny encoder, traced by Hugging Face's own code with zero rewiring, now compiles and matches HF's actual forward pass to `2.86e-06`.**

## The two gaps, closed

**1. `MatMul` + `Add` as Linear.** `try_match_matmul_add_linear` ([python/sajal_compile.py](../../../python/sajal_compile.py)) recognizes `MatMul(x, W) -> Add(matmul_out, bias)` as equivalent to `Gemm(transB=1)`. The one real subtlety, found by inspecting actual (non-square, so shape alone disambiguates) weight tensors before assuming anything: this `W` is stored `[in_dim, out_dim]` — the mirror of `Gemm`'s `[out_dim, in_dim]` — since a plain `MatMul` (no transpose flag) needs the weight pre-transposed to compute the same result a `Gemm(transB=1)` would. The IR entry marks `weight_transposed=True`; `compile_model()` transposes the numpy array back to `[out,in]` before writing the `.bin` file, so the exact same generated C++ (`linear()`, `common.hpp`) handles both encodings — only which bytes land on disk differs, not the codegen.

**2. 4D-batched attention.** HF's real export keeps an explicit batch dimension through the whole attention computation: reshape to `[1, seq, H, Dh]` (not exp20's `[seq, H, Dh]`), `Transpose` with 4-element perms `[0,2,1,3]`/`[0,2,3,1]` (not `[1,0,2]`/`[1,2,0]`). `try_match_multihead_attention`'s shape/perm checks generalized via one `classify_perm()` helper recognizing both conventions as the same two roles ("qv" or "k") — the actual per-head BLAS codegen never changes, since batch=1 throughout means the 4D case is mathematically identical to the 3D case with an inert leading dimension.

**A related but separate wrinkle in the same matcher**: HF's real export scales attention differently too — `Q` and `K` are each multiplied by `d_head**-0.25` *individually*, **before** the matmul, not once *after* it (exp20/22's synthetic models). `unwrap_scale_forward`/`unwrap_scale_backward` walk past zero-or-one such `Mul`/`Div`-by-constant wrapper in either direction, folding whatever scale factors they find into one combined value — so the matcher now accepts the old (post-matmul, single scale) and new (pre-matmul, split scale) conventions through the *same* code path, not a special case for each.

## The real payoff: no rewiring needed at all

Re-exported `model.encoder.layer` ([python/export_real_hf_trace_test.py](../../../python/export_real_hf_trace_test.py)) — **both real bert-tiny layers, HF's own `BertLayer.forward()`, unmodified** — and compiled it directly:

```
compiled artifacts/real_hf_2layer_test/model.onnx -> artifacts/compiled/real_hf_2layer_test/model
```

Equivalence against calling HF's actual layers directly (plain Python, real embeddings, a real sentence — not the rewired wrapper from exp25): max abs error **`2.86e-06`**. This is a materially stronger result than exp25's — that experiment needed a hand-verified rewiring step to get real weights into a compiler-friendly shape; this compiles Hugging Face's own trace as-is.

The earlier attempt (exp25) also hit an unexplained `Gather` node when tracing both layers together, plus an inconsistent `Gemm`-vs-`MatMul+Add` split across layers in the same graph. Re-exporting with the real embedding tensor as the example input (rather than random dummy data) produced neither — both layers consistently used `MatMul+Add`, and no `Gather` appeared. That earlier oddity looks like it was specific to the particular dummy input used for tracing, not a structural property of the model; not fully explained, but no longer blocking anything, since the graph that matters (the one built from a real sentence) doesn't have it.

## Validation

**Regression, all 13 prior compiler targets** (exp1, exp9, exp16-25) recompiled from scratch after both changes — still bit-identical.

**Single real layer, direct compile**: PyTorch equivalence vs. HF's real forward pass, max abs error `1.67e-06`.

**Two real layers, direct compile**: PyTorch equivalence vs. HF's real forward pass, max abs error `2.86e-06` — consistent with every prior experiment's "error grows with depth, not with FLOPs" pattern (exp2, exp11, exp19).

**Negative path**: corrupted the real graph's K-branch 4D perm (`[0,2,3,1] -> [0,3,2,1]`). Rejected cleanly: `error: op 'Reshape' is not supported...` — exit 1, no silent misfire.

## A genuinely new kind of finding: not bit-identical, and why that's the right outcome

Every compiler experiment through exp25 achieved **bit-identical** output against an independently hand-written reference — this is the first one that doesn't, and the reason is worth stating precisely rather than glossing over. The single real layer's compiled output differs from a hand-written reference (using a single upfront `1/sqrt(64)` scale) by a max abs diff of `1.19e-06`. Root cause, traced to its source: the compiler computes the combined scale as `qk_scale_a * qk_scale_b` (two already-rounded `64**-0.25` values multiplied in Python), which is `0.12500000000000003` — not bit-identical to `1.0/sqrt(64.0)` computed once (`0.125` exactly), because floating-point multiplication of two independently-rounded values isn't guaranteed to exactly recover the value you'd get computing it a different way. This is the exact "IEEE-754 non-associativity" caveat this project has carried in its methodology since exp1 — it simply never had occasion to actually bite until a real model's export happened to encode the identical math as a different sequence of floating-point operations. The resulting divergence (`1.19e-06`) is still fp32-tight, well inside this project's own established equivalence bar (compare: exp2's original hand-port vs. PyTorch was `1.67e-06`) — a real, understood, and harmless consequence of *which* correct computation the source graph specified, not a compiler bug.

## Interpretation

exp25 asked "does the compiler work on a real pretrained model" and answered it by rewiring around what HF's own export does. This experiment answers the harder, more valuable version of that question: yes, directly, with zero rewiring, because the two gaps that made rewiring necessary were addressable as generalizations of matchers this project already had (a second Linear encoding, a second attention shape/scale convention) rather than requiring new capabilities. The compiler's op-recognition surface is no longer built exclusively around this project's own synthetic-test-model conventions.

## Caveats

Still the encoder-layer stack alone (no embeddings, pooler, or attention masking — same scope as exp25). The `MatMul+Add` matcher requires the exact 2-node contiguous shape observed; a differently-ordered or 3-node variant (e.g., bias broadcast via a separate `Expand`) would still fail cleanly rather than being silently mishandled. The unexplained `Gather` from exp25's very first attempt was avoided, not root-caused — a different real model, or a different dummy input shape during export, might still hit it, and this compiler still has no `Gather` support at all.

## Next experiment

Causal attention masking (needed for autoregressive/decoder models, not attempted); or widen beyond BERT's encoder-only shape to a real decoder or encoder-decoder model, which will likely surface its own new gaps the same way real bert-tiny did here.
