# Experiment 12 — Resolving the Warm-Latency Nuance: Width, Not Depth (or Toolchain)

**The open question from [exp11](../exp11-real-pretrained-transformer/results.md)**: real bert-tiny (hidden=128, 2 layers) retained a modest native warm-latency edge (1.25x vs. ONNX Runtime) while exp2's synthetic transformer (hidden=256, 1 layer) collapsed to ~1.0x — despite bert-tiny making *more* total BLAS calls (twice the layers). exp5's "operator count explains it" theory predicted the opposite. But that comparison had two uncontrolled confounds: exp2 used `transformer_model.hpp`/the legacy ONNX exporter; exp11 used `bert_model.hpp`/the newer dynamo exporter — either could explain the difference as easily as architecture shape.

**Method**: hold the toolchain fixed (every config here uses `bert_model.hpp` and `bert_common.py`'s exporter — exp11's exact toolchain, not exp2's) and vary only `hidden_size` (128 vs. 256, matching bert-tiny vs. exp2's widths) and `num_layers` (1 vs. 2, matching exp2 vs. bert-tiny's depths) independently, across all 4 combinations. `head_dim=64` held fixed throughout (matches both exp2 and exp11 exactly — `num_heads = hidden/64`). `seq_len=12` fixed (matches exp11's actual benchmark case). Random weights (this is a pure latency ablation, no accuracy claim). Code: [bert_width_depth_sweep.py](../../../python/bert_width_depth_sweep.py). Equivalence spot-checked on the most complex config (hidden=256, 2 layers) before trusting the numbers: max abs error 1.19e-06, still fp32-tight.

## Results

| hidden_size | num_layers | native p50 | onnx p50 | native vs. onnx |
|---:|---:|---:|---:|---:|
| 128 | 1 | 0.05279ms | 0.07412ms | **1.40x** |
| 128 | 2 | 0.10671ms | 0.12281ms | **1.15x** |
| 256 | 1 | 0.13333ms | 0.13204ms | **0.99x** |
| 256 | 2 | 0.28771ms | 0.25210ms | **0.88x** |

Full data: [results.json](results.json).

## Width is the dominant variable — and this also resolves the toolchain confound

**At hidden=128, native retains an edge regardless of depth** (1.40x at 1 layer, 1.15x at 2 layers). **At hidden=256, native has no edge regardless of depth** (0.99x at 1 layer, 0.88x at 2 layers) — and that 0.99x, produced through exp11's completely different toolchain, lands almost exactly on exp2's originally-measured ~0.85-1.11x range (research/experiments/exp3-size-sweep). **That match is the resolution to the confound worry**: if the toolchain (exporter version, `bert_model.hpp` vs. `transformer_model.hpp`) were driving exp2 vs. exp11's difference, reproducing exp2's shape through exp11's toolchain should NOT have reproduced exp2's number — but it did. The difference was never about which exporter or which C++ header was used. It's architecture shape, specifically width.

**Depth has a real but secondary effect**, consistently shifting the ratio in ONNX Runtime's favor (1.40x→1.15x at hidden=128; 0.99x→0.88x at hidden=256) — more layers means more total BLAS calls, consistent with exp5's mechanism, but as a compounding factor on top of width, not the primary driver.

## Why width, specifically

`head_dim` (64) was held constant, so this isn't about attention's per-head matmul size — exp5 already characterized that regime. What changes with `hidden_size` is the size of the **linear projection layers**: Q/K/V/output-projection are `[H,H]` matrices and the FFN is `[H,4H]`/`[4H,H]` — both scale as O(H²) in FLOPs. At `hidden=128` these are small matrices (128×128, 128×512); at `hidden=256` they're 4x larger in FLOPs (256×256, 256×1024). exp5's own microbenchmark already showed the mechanism this predicts: its "linear_proj" shape test (`M=32,K=256,N=256` — matching `hidden=256`'s scale) found *no* batching/call-count benefit, because compute already dominated dispatch overhead at that size; its "attn_head" shape (small, dispatch-dominated) showed a large benefit. This experiment shows the same threshold applies to the *linear layers themselves*, not just attention: below some width, the whole forward pass (attention dispatch AND linear layers) stays dispatch-overhead-dominated, favoring native; above it, the linear layers alone have enough real compute to erase native's edge, matching exp2's original result.

## Interpretation

This upgrades exp5's "operator count, not language" theory into a more complete one: **"total compute relative to fixed per-call dispatch overhead" — driven primarily by model width, secondarily by depth/call-count — determines whether native retains a warm-latency edge.** exp1's tiny MLP (in=64,hidden=128) and bert-tiny (hidden=128) both sit in the narrow, dispatch-dominated regime where native wins something real; exp2-8's synthetic transformer (hidden=256) and this sweep's hidden=256 configs sit in the wider, compute-dominated regime where it doesn't. This is now a testable, falsifiable rule rather than an open nuance — and it was reached by holding the toolchain fixed and varying one architectural variable at a time, exactly the kind of controlled ablation exp3's original (confounded) sweep skipped.

## Caveats

Only two points per axis (128 vs. 256 width; 1 vs. 2 layers) — a real threshold curve (where exactly does the crossover happen between 128 and 256?) isn't mapped, only bracketed. Single hardware (Apple M4), single seq_len (12). Random weights only — this ablation is about latency, not correctness, by design.

## Next experiment

Map the actual width threshold: sweep `hidden_size` more finely (e.g. 128, 160, 192, 224, 256) at a fixed depth to find where `native_vs_onnx` crosses 1.0x — turning "somewhere between 128 and 256" into a specific, actionable number for what counts as "small enough" when designing future Sajal Labs model artifacts.
