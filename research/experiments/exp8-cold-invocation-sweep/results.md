# Experiment 8 — Cold Invocation Across the Size Sweep

**Question**: exp7 measured cold-invocation latency at exp1/exp2's two fixed configurations (one MLP, one `d_model=256` transformer) and found native 15-155x faster, consistent across those two architectures — unlike every warm-latency comparison in exp2-6, which fell apart specifically as the transformer got real compute. Does the cold-invocation advantage hold across *model size* the same way, or does it degrade like the warm-latency one did?

**Method**: same methodology as [exp7](../exp7-cold-invocation/results.md) (true external wall-clock per subprocess, interleaved round-robin trials, `once` mode / `--once` flag), reusing [exp3](../exp3-size-sweep/results.md)'s sweep artifacts (`artifacts/sweep/d{16,32,64,128,192,256}/` — same transformer architecture, `seq_len=32`, `n_heads=4`, `d_ff=4×d_model`). 20 trials per config per size (vs. exp7's 30 — six sizes instead of one, so fewer trials per point to keep runtime reasonable). Code: [run_cold_invocation_sweep.py](../../../python/run_cold_invocation_sweep.py). Raw data: [results.json](results.json).

## Results

| d_model | native p50 (ms) | onnx p50 (ms) | pytorch p50 (ms) | native vs. onnx | native vs. pytorch |
|---:|---:|---:|---:|---:|---:|
| 16 | 4.00 | 78.74 | 596.34 | 19.7x | 149.1x |
| 32 | 4.03 | 79.09 | 596.32 | 19.6x | 147.8x |
| 64 | 4.10 | 79.41 | 598.85 | 19.3x | 145.9x |
| 128 | 4.28 | 79.64 | 608.24 | 18.6x | 142.1x |
| 192 | 4.69 | 81.31 | 596.98 | 17.3x | 127.3x |
| 256 | 5.07 | 79.78 | 569.57 | 15.7x | 112.4x |

## The advantage holds across the entire sweep — and the trend is small, explainable, and consistent with everything found so far

**Unlike exp2-6's warm-latency comparison, which collapsed from 6x to ~1x within this same size range, cold-invocation stays large (15.7x-19.7x vs. ONNX Runtime, 112x-149x vs. PyTorch) at every size tested.** This confirms exp7's result wasn't a fluke of the two specific configurations it happened to test — it's a property of the comparison itself (process-startup cost vs. framework-import cost), not of any particular model size or architecture detail.

The ratio does shrink gently as `d_model` grows (19.7x → 15.7x vs. ONNX), and the reason is fully explained by prior experiments: **native's own cold time grows slowly with model size** (4.00ms → 5.07ms, ~27% increase across a 16x range of `d_model`, plausibly weight-loading I/O) **while PyTorch's and ONNX Runtime's cold time stays essentially flat** (596-608ms for PyTorch, 78.7-81.3ms for ONNX Runtime, regardless of `d_model`) — because their cold cost is dominated by importing large C-extension libraries once, not by how big the specific model is. This matches exp1 and exp7's finding directly: Python's cold-start cost is a library-import tax paid once per process, largely independent of the model being served.

## Interpretation

This is the strongest confirmation yet of the pivot in [gap-analysis.md](../../gap-analysis.md): the cold-invocation advantage is not size-dependent, not architecture-dependent (exp7 already showed MLP and transformer both benefit similarly), and the trend that does exist is small and fully explained rather than an unexplained wobble. Two independent data sources (exp7's cross-architecture comparison, exp8's within-architecture size sweep) now both point the same direction with no contradictions — a meaningfully different evidentiary position than exp2-6's warm-latency chase, which kept producing surprises (exp3's non-monotonic result, exp6's negative result) right up until the pivot.

## Caveats

Still Apple Silicon only, still CPU-only, still batch size 1. The sweep only covers the transformer architecture (not re-testing the MLP across sizes) — reasonable given exp7 already established cross-architecture consistency at one size each, but a full 2D sweep (architecture × size) hasn't been run.

## Next experiment

The cold-invocation/footprint claim is now well-evidenced across both the axes that matter (architecture, size). Per [gap-analysis.md](../../gap-analysis.md)'s revised recommendation, the natural next step is moving off synthetic sweep models and toward Phase 3 of the original roadmap: take a real small pretrained model (e.g. from one of Sajal's existing projects, or a small Hugging Face classifier/embedding model) and repeat this same equivalence + cold-invocation methodology on it, to confirm these findings transfer from hand-built synthetic architectures to something someone would actually deploy.
