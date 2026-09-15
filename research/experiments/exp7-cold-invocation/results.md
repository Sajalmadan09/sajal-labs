# Experiment 7 — Cold Invocation: The Pivoted Value Proposition

**Why this experiment**: exp1-6 chased warm per-call latency as the headline metric and it didn't hold up — native C++ never beat ONNX Runtime CPU on warm latency for any transformer-shaped workload (exp2-6), only exp1's trivial 2-matmul MLP showed a real win there. What held up in *every* experiment, at every size, regardless of architecture: cold-start and dependency-footprint advantages. This experiment measures that directly and rigorously, as the now-primary claim, rather than as a side note to a warm-latency comparison that didn't pan out.

**What "cold invocation" means here, and why it's different from exp1-6's "cold_start_ms"**: exp1-6 measured cold start *from inside* the process (a clock started at the top of `main()`/the script). That misses OS-level process creation, dynamic linker work, and process teardown — the parts of the cost an external caller (a serverless platform, a shell script, a scheduler) actually pays. This experiment times the *entire subprocess* from outside, wall-clock, for one process that does exactly one inference and exits — [native `once` mode](../../../native/mlp.cpp), [Python `--once` mode](../../../python/bench_pytorch.py). 30 trials per configuration, **interleaved round-robin** (not run back-to-back per config) to avoid one config's runs warming the OS page cache for a later config sharing the same shared libraries. Code: [run_cold_invocation_bench.py](../../../python/run_cold_invocation_bench.py). Raw data: [results.json](results.json).

## A bug found and fixed along the way — left visible, not edited out

Building this experiment surfaced a real bug in exp1: `bench_onnx.py` imported `IN_DIM` from `tiny_mlp.py`, which imports `torch` at module scope — so exp1's "ONNX Runtime" cold-start and memory numbers had *always* been silently paying PyTorch's import cost too. First sign of trouble: this experiment's initial run showed `onnx_transformer` at 77ms but `onnx_mlp` at 623ms — the same library, wildly different cold cost. Interleaving trial order (to rule out an OS-cache ordering artifact) didn't fix it, which pointed at a real code difference rather than a benchmark-methodology one. Tracing it down found the accidental torch import. Fixed by reading `in_dim` from `shapes.txt` directly, matching the pattern `bench_onnx_transformer.py` already used (it never had this bug). **[exp1's results.md](../exp1-tiny-mlp/results.md) has been corrected accordingly** — cold start from ~431-479ms to ~48ms, peak RSS from ~238MB to ~58MB; warm per-call latency was unaffected. This is exactly the "don't silently patch it, explain what failed and why" principle in practice.

## Results

| Config | Cold invocation p50 (ms) | Cold invocation p95 (ms) |
|---|---:|---:|
| **native_mlp** | 3.85 | 4.68 |
| onnx_mlp | 78.50 | 104.04 |
| pytorch_mlp | 598.70 | 712.65 |
| **native_transformer** | 5.05 | 8.86 |
| onnx_transformer | 80.30 | 87.05 |
| pytorch_transformer | 605.53 | 709.17 |

| Ratio (native vs.) | MLP | Transformer |
|---|---:|---:|
| vs. ONNX Runtime | **20.4x faster** | **15.9x faster** |
| vs. PyTorch eager | **155x faster** | **120x faster** |

Dependency footprint, unchanged from prior experiments and included here for completeness: native binaries 0.04-0.06MB vs. torch (535.78MB on disk) and onnxruntime (78.28MB on disk) site-packages.

## Interpretation

**This is the strongest, cleanest result in the project so far, and it holds regardless of model architecture** — the MLP and transformer ratios are close to each other (20.4x/15.9x vs. ONNX Runtime, 155x/120x vs. PyTorch), unlike the warm-latency comparisons in exp2-6 which fell apart specifically for the transformer. That consistency across architectures is exactly what exp2-6's warm-latency numbers never achieved, and it's the evidence base for the pivot: **for single-shot invocation — the shape of a serverless function call, a CLI tool, or an intermittently-invoked edge device — native C++'s advantage is large, consistent, and doesn't depend on getting kernel-level optimization details right** (unlike the warm-latency chase in exp3-6, which required correctly diagnosing dispatch overhead, buffer reuse, and BLAS call counts just to stay roughly at parity).

This reframes RQ5 ("when does native provide a meaningful advantage") with real, repeated evidence: **the advantage is in avoiding Python/framework process startup cost, not in out-computing a mature runtime's kernels.** That's a narrower, more honest, and more defensible claim than "native inference is faster" — and per research/gap-analysis.md, it was always the more likely source of advantage; this experiment is the first time it's been measured as the primary, standalone claim rather than a footnote to warm latency.

## Caveats

Still one machine (Apple Silicon), still tiny models, still batch size 1. `time.perf_counter()` wrapping `subprocess.run()` includes Python's own subprocess-spawning overhead (fork/exec from the orchestrator process) equally for all six configs, so it shouldn't bias the *comparison* between them, but it means the absolute native numbers here (3.85-5.05ms) are not "true OS process creation time" in isolation — a lower-level measurement (e.g. `hyperfine`, or timing from a shell rather than Python) might show the native side even faster in absolute terms. The very first invocation after a fresh build showed an anomalous ~530ms spike (disk-cache-cold for a just-linked binary) — not included here since all 30 trials per config ran with warm disk caches, representative of repeated/production invocation, not a machine's very first boot.

## Next experiment

This result is strong enough to treat as the project's headline claim going forward. The natural next step is broadening evidence for it rather than optimizing further: run this same cold-invocation harness across the exp3 size sweep (`d_model` 16-256) to confirm the ~15-20x ONNX Runtime advantage holds across sizes, not just at exp1/exp2's two fixed configurations.
