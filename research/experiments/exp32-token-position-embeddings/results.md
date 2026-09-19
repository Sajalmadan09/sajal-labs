# Experiment 32 — Token and Position Embeddings via Gather

**Why this experiment**: exp28-31 built real GPT-2's decoder stack entirely, but every one of those experiments started from *pre-computed* float embeddings — the input to the compiled model was always `wte(input_ids) + wpe(position_ids)`, already run in Python. This closes that gap: the compiled model now takes raw integer token ids as its own external input and does the embedding lookup itself.

**The result**: closed, but the natural "one combined graph, embeddings feeding straight into 12 real transformer blocks" version doesn't work the way exp28-31 might suggest — not because embeddings are hard, but because the ONLY way to get a genuine (non-constant-folded) position-embedding lookup makes the *entire* rest of the graph dynamically-shaped, which is a much bigger, different problem this experiment deliberately didn't try to solve. The right-sized answer: two independently compiled, independently validated artifacts, chained.

## The first real finding: position embeddings usually aren't there at all

Exporting `wte(input_ids) + wpe(torch.arange(seq_len))` at a **fixed** sequence length — every prior experiment's convention — showed something worth stating plainly: PyTorch's exporter **constant-folds the entire position-embedding computation into a precomputed bias tensor**. No `Gather` for position ids at all; just `Gather(wte.weight, input_ids)` followed by `Add` against a baked-in `[seq_len, hidden]` constant. This makes complete sense once seen (a fixed `arange(32)` is deterministic, so the exporter just evaluates it once) but wasn't assumed going in — it was found by exporting and reading the graph, the same discipline as every prior experiment.

Forcing `dynamic_axes` on the sequence-length dimension recovers the real computation: `Shape(input_ids) -> Gather(shape, 0) -> Cast -> Range(0, len, 1) -> Gather(wpe.weight, range_out)`. This is the pattern this experiment actually targets — but it comes at a cost, described below.

## The second finding: dynamic shape isn't just an embeddings problem

Combining the (dynamically-shaped) embedding layer with the (previously fixed-shape) 12-layer transformer stack in **one** exported graph produces 2,410 nodes — nearly triple exp31's 900, almost none of it embeddings. Making `input_ids`'s length genuinely dynamic propagates shape uncertainty into every one of the 12 blocks' own internal reshapes (their head-split `Reshape`s, previously fed literal `[1, seq, H, Dh]` constants, now need runtime `Shape`/`Concat`-built shapes instead), producing 240+ `Unsqueeze`/`Squeeze` pairs and a comparable explosion in `Shape`/`Slice`/`Concat` bookkeeping repeated per layer. Supporting that fully would mean teaching the compiler to read shapes computed at the ONNX-graph level rather than always from literal constants — a real, much larger capability than "recognize an embedding lookup," and out of scope here.

## What actually got built

Two new matchers ([python/sajal_compile.py](../../../python/sajal_compile.py)): `try_match_token_embedding` recognizes `Gather(embedding_table, input_ids)` where `input_ids` is literally the graph's own declared input — the one part of a real forward pass that genuinely can't be constant-folded, since actual token ids vary every call. `try_match_position_embedding_chain` recognizes the whole `Shape->Gather(idx 0)->Cast->Range->Gather` chain and — since this compiler already knows `n` at runtime as `forward()`'s own parameter — throws away everything except the final table lookup, which becomes "copy the position table's first `n` rows," no indices needed at all. Both are triggered the same way exp29's `Split` lookahead and exp30's dead-chain marking were: the position chain's dependencies (`Shape`) sit *before* the node that actually claims them, so it needs a lookahead rather than the usual defer-after-match pattern.

The bigger change: this is the **first time the compiled model's own external input type changes** — `forward()` takes `const std::vector<int64_t>& input_ids` instead of `const std::vector<float>& X` for any graph starting with an embedding op. `run_mode`/`bench_mode`/`once_mode` and the driver's dummy-input construction all branch on `has_embedding` the same way they already branch on `has_attention` (exp28). Two new `common.hpp` primitives: `gather_embedding_rows` (real, data-dependent lookup) and `position_embedding_rows` (literally `std::copy` of the table's first `n` rows, since the indices are always `0..n-1` by construction).

A real, if minor, bug surfaced immediately: `output_dim`'s "find the last `Linear`" fallback had no fallback at all for a graph with *no* `Linear` (an embedding-only model — token lookup, position lookup, add, nothing else). Fixed by falling back to the final IR entry's own dimension.

## Validation

**Regression, all 21 prior compiler targets** (exp1, exp9, exp16-31) recompiled from scratch — still bit-identical.

**Real GPT-2's embedding layer** ([python/export_real_gpt2_embeddings_test.py](../../../python/export_real_gpt2_embeddings_test.py), real `wte`/`wpe` weights, dynamic-axis export): compiles, generated code reads `input_ids.data()` directly (no `var()` float lookup) and calls `gather_embedding_rows`/`position_embedding_rows` as designed. Equivalence vs. real GPT-2's own `wte(input_ids) + wpe(arange(seq))`: **exactly `0.0`** max abs error — expected and unsurprising, since a table lookup plus one elementwise add involves no accumulated floating-point rounding at all, unlike every matmul-heavy result in this project's history.

**Chained end-to-end pipeline**: fed this experiment's compiled embedding artifact's own output directly into exp31's already-compiled 12-layer stack (two independently produced binaries, no shared ONNX graph) and compared against calling real GPT-2 fully end-to-end in Python (tokenize → `wte`+`wpe` → all 12 blocks, no ONNX involved at any point). Max abs error `1.83e-04` — essentially identical to exp31's own number, since the embedding step itself contributes zero error. **This is genuinely "raw token ids in, real GPT-2 hidden states out," using only compiled artifacts, matching real GPT-2's own answer.**

**Negative path**: corrupted the `Range` node's `delta` constant from `1` to `2` (a stride-2 position sequence instead of `0,1,2,...`). Rejected cleanly (`error: op 'Shape' is not supported...`) — the malformed chain no longer matches the recognized pattern, so the whole thing falls through rather than silently computing wrong positions.

## Interpretation

The honest scope decision here mirrors exp25's original one for BERT: something real was found to be bigger than the original ask (dynamic-shape propagation, not "recognize `Gather`"), and rather than either ignoring it or absorbing it wholesale, the tractable piece was closed directly and validated end-to-end through composition instead of forcing everything into one ONNX graph. The chained-artifact result is not a lesser achievement than a single combined graph would have been — every number a downstream user cares about (real weights, real architecture, real end-to-end output) is identical either way; only the packaging differs.

## Caveats

Position embeddings are only recognized in their dynamic, `Range`-based form — the far more common fixed-shape case (a constant-folded per-position bias, what real GPT-2 naturally produces at any *fixed* sequence length, which is every prior experiment's own convention) is not handled, since a constant bias tied to one specific sequence length would silently break this compiler's existing "works for any `n` at runtime" property. True end-to-end dynamic shape support (reading `Shape`/`Concat`-computed reshape targets instead of only literal constants) remains unattempted and would be a substantially larger effort.

## Next experiment

Either the `lm_head` projection (finishing the other end of the pipeline — hidden states back to vocabulary logits, likely straightforward given it's just another `Linear`); or, more ambitiously, genuine dynamic-shape support, which this experiment found is the real prerequisite for combining embeddings and a transformer stack into one compiled graph rather than two chained ones.
