# Sajal Labs

Native AI inference research: can trained ML models be transformed into
minimal, dependency-light C/C++ inference artifacts without losing
predictive fidelity — and when does that actually beat Python-based serving?

**Philosophy:** train anywhere, deploy natively, measure everything. No
claim that C++ is inherently faster — every advantage is a hypothesis to
benchmark, not an assumption.

## Status (2026-09-16)

Phase 0 (landscape research), Phase 1 (educational prototype), and the real
pretrained model in Phase 3 are done — [research/](research/) has the
ecosystem survey, academic literature review, and gap analysis;
[research/experiments/](research/experiments/) has nine completed
experiments (tiny MLP → transformer block → size sweep → three rounds
chasing a warm-latency mystery → the pivot → confirmed across the size
sweep → confirmed on a real trained model).

**Current value proposition** (revised after experiments, see
[gap-analysis.md](research/gap-analysis.md)'s "Revised recommendation"
section): native C++'s advantage is in **cold/single-shot invocation
latency and dependency footprint** (15-155x faster process-to-answer time,
~10-40x smaller memory/disk footprint, consistent across model
architectures) — **not** warm-server throughput, where it did not beat
ONNX Runtime CPU for any transformer-shaped model tested. Target use case:
serverless functions, CLI tools, intermittently-invoked edge devices —
not sustained high-throughput serving.

## Published models

- [sajalmadan09/indian-name-gender-classifier](https://huggingface.co/sajalmadan09/indian-name-gender-classifier) — exp9/10's real gender classifier, weights in native/safetensors/ONNX formats, model card with honest accuracy/limitations/benchmark data. Local package source: [huggingface/gender-classifier/](huggingface/gender-classifier/) ([python/export_hf_package.py](python/export_hf_package.py) regenerates it from the experiment data).

## CLI

```bash
cd native && make sajal
./sajal inspect <artifacts_dir>          # detects model kind, shows dims/vocab/size
./sajal run <artifacts_dir> [text]       # single prediction; text required for text-mlp kind
./sajal bench <artifacts_dir>            # warmup + 500-iteration timed loop, same format as mlp/transformer's bench mode
```

Model "kind" (`mlp`, `transformer`, or `text-mlp`) is detected from what's already in the artifacts directory — a `vocab.txt` means text input (exp9/10's shape), `shapes.txt`'s field count distinguishes plain MLP (exp1's shape) from transformer (exp2-8's shape) — no new manifest format invented for this.

`sajal` links the same model headers (`mlp_model.hpp`, `transformer_model.hpp`, `text_features.hpp`) directly and does inference in-process, rather than shelling out to `mlp`/`transformer`/`gender_predict` — those stay exactly as they are as dedicated, minimal, unchanged binaries, since exp1-10's benchmark numbers depend on them not having a dispatch layer. Verified empirically that this design choice matters: `sajal run` cold-invocation latency (2.90ms p50) matches the dedicated `gender_predict once` binary (2.94ms p50) almost exactly, despite the larger linked binary — shelling out would have doubled process-startup cost and undermined the very thing ten experiments were measuring.

## Experiments

1. [Tiny MLP](research/experiments/exp1-tiny-mlp/results.md) — equivalence + benchmark harness established
2. [Tiny transformer](research/experiments/exp2-tiny-transformer/results.md) — warm-latency edge nearly vanishes at scale
3. [Size sweep](research/experiments/exp3-size-sweep/results.md) — crossover isn't about FLOPs
4. [Buffer reuse](research/experiments/exp4-buffer-reuse-sweep/results.md) — ruled out as the cause
5. [GEMM dispatch overhead](research/experiments/exp5-gemm-dispatch-overhead/results.md) — isolates the real mechanism
6. [Naive attention](research/experiments/exp6-naive-attention-sweep/results.md) — negative result, reverted
7. [Cold invocation](research/experiments/exp7-cold-invocation/results.md) — the pivot, with a bug found and corrected along the way
8. [Cold invocation sweep](research/experiments/exp8-cold-invocation-sweep/results.md) — pivot confirmed across all 6 sizes
9. [Gender classifier](research/experiments/exp9-gender-classifier/results.md) — real trained model on real data, pivot holds
10. [End-to-end native pipeline](research/experiments/exp10-e2e-native-pipeline/results.md) — feature extraction ported to C++, raw string in / prediction out, zero Python

## Roadmap

0. Landscape research & gap analysis — done
1. Educational prototype (tiny NN in Python/C/C++) — done
2. Native runtime core (Tensor, Linear, ReLU, Softmax, LayerNorm, GELU) — done ([native/common.hpp](native/common.hpp))
3. Model equivalence (real trained model) — done (exp9: real Indian-name gender classifier, 100% prediction agreement)
4. Benchmarking vs. strong baselines — done, revised toward cold-invocation
5. Packaging — done (safetensors + ONNX + native formats, machine-readable benchmark data)
6. Developer experience (CLI) — done ([native/sajal.cpp](native/sajal.cpp): `inspect`/`run`/`bench`, in-process dispatch across model kinds)
7. Hugging Face integration — done ([sajalmadan09/indian-name-gender-classifier](https://huggingface.co/sajalmadan09/indian-name-gender-classifier), published)
8. Transformer support — architecture proven (exp2-8); real pretrained transformer not yet tried (exp9 used the MLP architecture)
9. Hardware optimization (SIMD/CUDA) — Apple Accelerate/AMX only so far; no CUDA hardware available
10. Research publication
