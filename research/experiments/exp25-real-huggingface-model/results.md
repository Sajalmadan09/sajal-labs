# Experiment 25 — A Real Pretrained Model from Hugging Face

**Why this experiment**: exp22-24 each found real gaps by testing against something more real than a synthetic model (exp2's own code, multi-block stacks, opset<20 exports) — but every test so far was still a PyTorch module written or wrapped by this project. This tries the compiler against an actual published model nobody in this project wrote: [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny), the same model exp11-14 hand-ported to C++.

**The result, honestly scoped**: tracing Hugging Face's own `BertLayer.forward()` directly turned out to be a much bigger, different problem than "does the compiler need a new op" — inspecting the graph first showed it uses several patterns this compiler was never designed around, matching modern `transformers`/`torch.onnx` internals rather than model architecture. Widening the compiler to accept HF's own trace as-is was out of scope for one experiment. Instead: the REAL pretrained weights, in the project's own already-supported attention convention (verified numerically faithful to HF's actual output), compile and run **bit-identical** to a hand-written reference — with zero changes to `sajal_compile.py`.

## What tracing HF's actual code produces, and why none of it was assumed

Exported `model.encoder.layer` (real bert-tiny: 2 layers, hidden=128, 2 heads, intermediate=512) directly, `dynamo=False`, and read the resulting graph before writing anything. Four real mismatches with what the compiler recognizes, all found by inspection rather than guessed at:

1. **Linear is `MatMul` + `Add`, not `Gemm`** — modern `transformers`/`torch.onnx` doesn't fuse the weight matmul and bias add the way the legacy synthetic-model exports (exp15-24) always did.
2. **Attention is 4D, not 3D** — HF's code keeps an explicit batch dimension throughout (`reshape` to `[1, seq, heads, head_dim]`, `Transpose` with 4-element perms like `[0,2,1,3]` and `[0,2,3,1]`), where every synthetic test model squeezed batch out entirely (3D reshape, 3-element perms).
3. **Split square-root scaling** — Q and K are each multiplied by `d_head**-0.25` *before* the `MatMul`, not scaled once *after* it (exp20/22's pattern) — a numerically-motivated but structurally different way of writing the identical math.
4. **Inconsistent op choice across layers in the same graph** — exporting *both* layers together produced `Gemm` for layer 1 but `MatMul`+`Add` for layer 0, and an unexplained `Gather` node between them. This is a `torch.onnx` export-pass quirk, not anything about the model.

This is exactly why exp11 hand-ported BERT to C++ instead of compiling it in the first place — and now, with a compiler in hand, it's clear that gap was never really about missing math support; it's that HF's own traced graph shape is a moving target across library versions and attention backends, orthogonal to whether the underlying computation is something this compiler already knows how to run.

## The well-scoped alternative: real weights, this project's own convention

Rather than chasing HF's exact trace shape (a much larger undertaking — four new capabilities, at least, none of them about model architecture), [python/export_real_bert_tiny_test.py](../../../python/export_real_bert_tiny_test.py) copies the real pretrained weights (`query.weight`, `key.bias`, `LayerNorm.weight`, all 16 tensors per layer, verbatim from `model.encoder.layer[i]`) into a small wrapper using the exact same attention convention as exp20/21/23's synthetic test models — single combined scale, 3D reshape, batch-squeezed `[seq, hidden]` input.

**Verified this is the same model, not a different one**, before treating it as ground truth: ran the wrapper on real embeddings (bert-tiny's own tokenizer, a real sentence) and compared against calling HF's actual `model.encoder.layer[i]` directly (plain Python, no ONNX involved) — max abs diff **`2.44e-06`**, consistent with ordinary fp32 rounding, not a different computation.

## Validation

**Compiling**: `sajal_compile.py artifacts/real_bert_tiny_test/model.onnx ...` — succeeds with **zero changes to the compiler**. Every op this graph uses (`Gemm`, the multi-head attention pattern, `Add`, `LayerNormalization`, `Gelu`) was already recognized from exp16-23.

**Equivalence vs. PyTorch** (the rewired wrapper's own output): max abs error `2.62e-06` — combined with the `2.44e-06` diff against HF's real forward pass above, the total round-trip from "real HF weights" to "compiled C++ output" stays within ordinary fp32 accumulation, at hidden=128 — a genuinely larger scale than any synthetic compiler test (`d=16`) has exercised.

**Bit-identical** against a new hand-written reference ([native/real_bert_tiny_reference.cpp](../../../native/real_bert_tiny_reference.cpp), same per-head BLAS-slicing pattern as every prior reference) — `cmp` reports no difference.

**Warm-loop latency** (bench mode, real weights, seq_len=15): p50 `0.122ms`, cold start `1.49ms` — plausible numbers at this scale, though not compared against a frozen baseline the way exp22 did (no "exp2-equivalent" hand-tuned driver exists for this exact rewired architecture; the point of this experiment was correctness on real weights, not a new performance claim).

## Interpretation

The honest finding here is two-layered. First: "does the compiler support a real pretrained model" doesn't have a clean yes/no answer — it depends entirely on which of that model's several mathematically-equivalent expressions you mean. HF's own traced graph: no, and getting to yes would be substantial new work unrelated to model correctness. The same real weights, expressed the way this compiler already understands attention: yes, with zero changes, bit-identical to hand-written code. Second: this is not moving the goalposts — the equivalence check against HF's *actual* forward pass (not just against the rewired wrapper's own PyTorch output) is what makes this a legitimate "real model" claim rather than "a model that happens to look like ours." Every number a downstream user of `sajal-labs` cares about (the real trained weights, the real architecture dimensions) is preserved; only the *expression* of the attention computation changed, and that change was verified, not assumed.

## Caveats

This is the encoder-layer stack alone — no token embeddings, no pooler, no tokenizer (exp11/14 already cover those, separately, for this exact model). Attention masking was never exercised (the test sentence has no padding). Widening the compiler to accept the `MatMul`+`Add` Linear form and 4D-batched attention directly (so HF's own export could be compiled as-is, no rewiring) is real, scoped future work — named explicitly rather than implied, since exp22-24's whole throughline has been "testing against something real finds gaps a synthetic model can't," and this one found the biggest gaps yet without pretending they're already closed.

## Next experiment

Two honest directions from here: (a) actually widen the compiler to accept `MatMul`+`Add` as a Linear op and 4D-batched attention shapes, closing the gap this experiment deliberately routed around; or (b) treat "rewire real weights into a known-good convention, verify against the real model, then compile" as the project's answer for now, and move toward end-to-end packaging (a small CLI or script that does this rewiring automatically for a given Hugging Face model name) rather than deeper op-coverage work.
