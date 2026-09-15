# Experiment 13 — Pinpointing the Width Threshold

**Question**: [exp12](../exp12-warm-latency-width-depth/results.md) established that width (`hidden_size`), not depth or toolchain, determines whether native retains a warm-loop edge over ONNX Runtime — but only measured two points (128 → 1.40x edge, 256 → 0.99x, no edge). Where between them does the crossover actually happen, and is it a sharp threshold or a gradual slope?

**Method**: same toolchain and methodology as exp12 (`bert_model.hpp`, `bert_common.py`'s exporter, `seq_len=12`), fixed at `num_layers=1` to isolate width cleanly (exp12 showed depth's effect is secondary). Five widths — `64, 128, 192, 256, 320` — all exact multiples of 64 so `num_heads = hidden/64` stays an integer and `head_dim` stays exactly 64 at every point, same invariant exp12 held. Code: [bert_width_sweep.py](../../../python/bert_width_sweep.py) (reuses exp12's `bench_one()` directly — this is the same ablation, more points on one axis). Equivalence spot-checked at the new endpoint (hidden=320): max abs error 2.09e-07, cosine similarity 0.99999994 — trustworthy.

## Results

| hidden_size | native p50 | onnx p50 | native vs. onnx |
|---:|---:|---:|---:|
| 64 | 0.02171ms | 0.05013ms | **2.309x** |
| 128 | 0.05271ms | 0.07473ms | **1.418x** |
| 192 | 0.09156ms | 0.10817ms | **1.181x** |
| 256 | 0.13562ms | 0.13310ms | **0.981x** |
| 320 | 0.18796ms | 0.17077ms | **0.909x** |

Full data: [results.json](results.json).

## A smooth curve, not a cliff — crossover at hidden ≈ 250

The five points fall on a clean, monotonically decreasing curve: `2.31x → 1.42x → 1.18x → 0.98x → 0.91x` as width doubles from 64 to 320. **Linear interpolation between the two bracketing points (192 at 1.181x, 256 at 0.981x) puts the crossover at hidden ≈ 250** — extremely close to bert-tiny's own actual `hidden=128` sitting comfortably on the winning side, and to exp2's `hidden=256` sitting almost exactly at parity, which is consistent with exp2's own originally-measured numbers hovering right around 1.0x (0.85-1.11x across the exp3 sweep) rather than being cleanly on one side or the other.

**This is a gradual tradeoff, not a step function.** There's no discrete "on/off" width past which native's edge vanishes — it decays smoothly as the linear-projection layers' O(H²) compute grows relative to the fixed per-call dispatch overhead exp5 characterized. That's the expected shape for a "fixed cost vs. scaling cost" tradeoff (exp5's Part A size-sweep showed the same smooth-then-compute-dominated shape for a single GEMM call), and it's reassuring that the mechanism identified at the single-operator level in exp5 produces the same qualitative curve shape at the whole-model level here.

## Interpretation

This turns exp12's bracketed observation into an actionable number: **for this project's actual hardware/toolchain, a model narrower than roughly hidden≈250 (with head_dim=64) can expect a real, measurable native warm-latency advantage; wider than that, expect parity or a slight ONNX Runtime edge.** Every real model this project has tested sits meaningfully on one side or the other of that line — exp1's MLP (hidden=128 equivalent) and bert-tiny (hidden=128) both comfortably in the native-favors-native zone; exp2-8's synthetic transformer (hidden=256) sitting almost exactly on the line, explaining why its own results hovered so close to 1.0x rather than clearly favoring either side.

## Caveats

Five points, one hardware target (Apple M4), one `seq_len` (12), `num_layers=1` only (exp12 showed depth shifts the curve down somewhat — the ≈250 crossover is specific to single-layer models; a deeper model's crossover would sit lower). This is a measured curve on this machine's Accelerate/BLAS behavior, not a universal constant — re-measuring on different hardware (different CPU, different BLAS library) could shift the threshold.

## Next experiment

This line of investigation (why/when warm latency favors native) is now well-characterized: mechanism (exp5), primary driver (exp12), and a concrete threshold (exp13). Per research/gap-analysis.md, the more valuable next step is likely moving away from further latency micro-characterization and back toward the project's broader roadmap — e.g., porting WordPiece tokenization to C++ (exp11's suggested next step, for a fully end-to-end real-transformer pipeline) or exploring quantization now that full-precision baselines are this well understood across both the cold-invocation and warm-loop axes.
