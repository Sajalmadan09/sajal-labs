# Experiment 29 — Combined Q/K/V via Split

**Why this experiment**: exp28 found six real differences trying a real GPT-2 decoder block, closed three, and deliberately routed around one rather than closing it — Q/K/V produced by a single combined projection (`c_attn`, one `Linear` outputting `3*d`) sliced into three via ONNX `Split`, rather than three dedicated `Linear`s. This closes that gap directly.

**The result**: closed, but it took two separate bugs to get there — one structural (node-visit ordering), one a genuine, longstanding latent bug that simply never had occasion to matter before this experiment.

## The design: one buffer, three offsets, same trick one level up

`try_match_multihead_attention` already addresses per-*head* slices of a `[n, d]` buffer via a BLAS pointer-offset trick (`buffer.data() + h*d_head`, `lda=d`) — no physical splitting, established since exp20. Split-based combined QKV is the same idea one level up: Q/K/V are column-slices of one wider `[n, 3*d]` buffer (offsets `0`, `d`, `2*d`; `lda = 3*d` instead of `d`). `resolve_split_source` ([python/sajal_compile.py](../../../python/sajal_compile.py)) resolves any tensor back through an optional `Split` producer to `(shared_buffer, column_offset, buffer_width)` — `(tensor_name, 0, d)` unchanged for the ordinary case (a dedicated Linear per Q/K/V, every prior experiment). The multi-head codegen now threads this triple through instead of a bare tensor name, composing the two offset tricks (`buffer.data() + column_offset + h*d_head`, `lda = buffer_width`) rather than treating them as unrelated cases.

## Bug 1: Split sits before its own consumer, node-order-wise

Compiling even a clean, isolated combined-QKV test model (no masking, no other GPT-2 complications) failed immediately: `error: op 'Split' is not supported...`. The main parse loop visits nodes strictly in order; `Split` always appears *before* the `Reshape` that triggers `try_match_multihead_attention` (Split produces the tensor the Reshape consumes), so the loop rejects `Split` outright the first time it's seen — before the attention match that would consume it ever gets a chance to run. Every other node this compiler defers (via `pending_emit`/`consumed_skip`) is deferred *after* a match already succeeded at its natural trigger point; `Split` needed to be resolved *before* reaching that trigger.

Fixed with a lookahead: when the main loop hits an unconsumed `Split`, it checks whether any of its outputs feeds a `Reshape` whose multi-head match would claim this exact `Split` node — if so, that match is resolved right there (same deferred-emission bookkeeping the natural trigger would have done) and `Split` is skipped; otherwise it falls through to the ordinary unsupported-op rejection, unchanged.

## Bug 2: two producer-lookup helpers assumed single-output nodes

Even with the lookahead in place, the match still failed to find `Split` as `Q`'s producer at all. Both `resolve_dim`'s fallback and the general-purpose `find_producer` looked up a tensor's producer with `list(node.output) == [tensor_name]` — an equality check against a *one-element* list. Every op this compiler had ever needed to look up a producer for (`Gemm`, `Add`, `Transpose`, `Reshape`, `MatMul`, `Softmax`) has exactly one output, so this happened to work for eight experiments. `Split` is the first multi-output node either helper ever needed to find — its three-element `output` list can never equal a one-element list, so both lookups silently returned "not found" for every `Split`-produced tensor. Fixed by switching both to membership (`tensor_name in node.output`), which is exactly equivalent for single-output nodes and correct for multi-output ones.

Neither bug was found by reasoning about the design ahead of time — both were found by trying to compile the simplest possible test case and reading exactly where and how it failed.

## Validation

**Regression, all 18 prior compiler targets** (exp1, exp9, exp16-28) recompiled from scratch after every change — still bit-identical. This matters especially for the membership-check fix: it touches `find_producer`, which every multi-head attention match (causal or not, split or not, real or synthetic) depends on.

**Isolated combined-QKV attention** ([python/export_combined_qkv_test.py](../../../python/export_combined_qkv_test.py), one `Linear(d, 3d)` + `Split`, no masking): generated code addresses Q/K/V as `buffer.data() + {0,16,32} + h*4` with `lda=48` — exactly the intended mechanism, confirmed by direct inspection of the generated `.cpp`. PyTorch equivalence: max abs error `5.96e-08`. **Bit-identical** against a new hand-written reference ([native/combined_qkv_reference.cpp](../../../native/combined_qkv_reference.cpp)).

**Negative path**: corrupted the `Split`'s size list (`[16,16,16] → [16,16,8]`, breaking the shape/dim consistency the match requires). Rejected cleanly (`error: op 'Split' is not supported...`) — the mismatched slice size fails the match's own dimension check, so nothing is silently mis-sliced.

**Honest limit, checked directly**: re-attempted the fully-unrewired real GPT-2 trace (the one exp28 rewired around) after this fix. It still fails, but now on a *different*, separately-scoped gap — GPT-2's own additive-bias causal mask decomposition (`Equal`+`Where`+`Add`, distinct from exp27's `Cast`+`Where`), which was explicitly out of scope for this experiment. The `Split` mechanism itself is confirmed working (via the isolated test and the negative-path check); real GPT-2's fully-direct compile needs that separate mask-variant gap closed too.

## Interpretation

This is the second experiment in the compiler line (after exp23) where the fix was a real bug found through testing, not a new capability added by design — and it surfaced two bugs stacked on each other, the second one hiding behind the first. `find_producer`'s single-output assumption is the kind of thing that's easy to write once, reuse everywhere, and never notice is wrong until something with more than one output shows up — which took 29 experiments (and a real model) to happen.

## Caveats

Same scope note as every combined-projection case: only a `Split` with resolvable per-output sizes (from its second input or its `split` attribute) is recognized; an even/implicit split with neither would fail cleanly rather than being guessed at. GPT-2's own fully-unrewired trace still needs the additive-bias mask variant closed separately — this experiment closes the projection-shape gap, not the masking-shape gap exp28 also deferred.

## Next experiment

Close the additive-bias (`Equal`+`Where`+`Add`) causal mask variant GPT-2's real trace actually uses — the one remaining piece for compiling real GPT-2 fully unrewired, no Python-side weight rewiring at all.
