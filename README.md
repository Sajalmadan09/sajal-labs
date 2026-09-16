# Sajal Labs

GitHub: [Sajalmadan09/sajal-labs](https://github.com/Sajalmadan09/sajal-labs)

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
- [sajalmadan09/bert-tiny-native-cpp](https://huggingface.co/sajalmadan09/bert-tiny-native-cpp) — native C++ port of [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny) (a third-party MIT-licensed model, clearly credited as `base_model` — not ours, the port is). Exact-match tokenizer, fp32-tight encoder equivalence, full exp11-14 benchmark data. Local package source: [huggingface/bert-tiny-native/](huggingface/bert-tiny-native/) ([python/export_bert_hf_package.py](python/export_bert_hf_package.py)).

## CLI

```bash
cd native && make sajal
./sajal inspect <artifacts_dir>          # detects model kind, shows dims/vocab/size
./sajal run <artifacts_dir> [text]       # single prediction; text required for text-mlp kind
./sajal bench <artifacts_dir>            # warmup + 500-iteration timed loop, same format as mlp/transformer's bench mode
```

Model "kind" (`mlp`, `transformer`, `text-mlp`, or `bert`) is detected from what's already in the artifacts directory — a `config.txt` means a WordPiece transformer (exp11-14's shape, BERT's own export filename, checked first since it's unambiguous), otherwise `vocab.txt` means bigram-MLP text input (exp9/10's shape), otherwise `shapes.txt`'s field count distinguishes plain MLP (4 fields, exp1's shape) from a synthetic transformer block (5 fields, exp2-8's shape) — no new manifest format invented for this. `bert`-kind artifacts without a real tokenizer (exp12/13's synthetic width/depth sweep configs) support `bench` but cleanly refuse `run` rather than producing nonsense output.

`sajal` links the same model headers (`mlp_model.hpp`, `transformer_model.hpp`, `text_features.hpp`, `bert_model.hpp`, `wordpiece.hpp`) directly and does inference in-process, rather than shelling out to `mlp`/`transformer`/`gender_predict`/`bert` — those stay exactly as they are as dedicated, minimal, unchanged binaries, since exp1-14's benchmark numbers depend on them not having a dispatch layer. Verified empirically that this design choice matters, twice now: `sajal run` cold-invocation latency matches the dedicated binary almost exactly for both the gender classifier (2.90ms vs. `gender_predict once`'s 2.94ms) and the much larger real BERT model (10.98ms vs. `bert once_e2e`'s 10.85ms) — shelling out would have doubled process-startup cost and undermined the very thing this project spent fourteen experiments establishing.

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
11. [Real pretrained transformer](research/experiments/exp11-real-pretrained-transformer/results.md) — prajjwal1/bert-tiny (4.4M real params), fp32-tight equivalence on 10 real sentences, 582x cold-invocation win over PyTorch+transformers
12. [Width vs. depth ablation](research/experiments/exp12-warm-latency-width-depth/results.md) — resolves exp11's warm-latency nuance: width (hidden_size) drives whether native retains an edge, not depth or toolchain
13. [Width threshold](research/experiments/exp13-width-threshold/results.md) — pinpoints the crossover at hidden≈250 (this hardware/toolchain); smooth curve, not a cliff
14. [WordPiece in C++](research/experiments/exp14-wordpiece-native/results.md) — real BERT tokenizer ported, exact token match on 17 real sentences; native beats even lean-tokenizer ONNX by 8.4x, and tokenizer library choice alone swings Python cold-start 26x
15. [Minimal ONNX→C++ compiler](research/experiments/exp15-minimal-compiler/results.md) — the first real compiler, not a hand-port: auto-generated code from exp1/exp9's ONNX graphs is bit-identical to the hand-written reference, `sajal` CLI works on the output with zero modification, fails cleanly on out-of-scope graphs
16. [LayerNorm + GELU in the compiler](research/experiments/exp16-layernorm-gelu/results.md) — generalized codegen (IR-driven, not a fixed template) to 5 ops; still bit-identical to a hand-written reference; caught and avoided a real correctness trap (reusing exp15's artifact convention would have let `sajal` silently misread the new graph shape)
17. [DAG support](research/experiments/exp17-dag-support/results.md) — the IR became a real graph (named tensors, not "previous op's output"); compiled a genuine residual connection (`LayerNorm(x + sublayer(x))`, the actual FFN half of a transformer block) bit-identical to a hand-written reference; exercised a validation check exp16 had flagged as unreachable
18. [Single-head attention](research/experiments/exp18-single-head-attention/results.md) — fused a 5-node ONNX pattern (Transpose→MatMul→Mul→Softmax→MatMul) into one IR op, reusing the exact BLAS trick already validated in transformer_model.hpp/bert_model.hpp; bit-identical to a hand-written reference; correctly rejects a malformed near-attention graph rather than mis-matching it
19. [Full encoder block](research/experiments/exp19-full-encoder-block/results.md) — combined residual `Add` + fused attention into one complete transformer block (post-norm attention+FFN, exp2's real architecture minus multi-head); zero compiler changes needed, bit-identical to a hand-written reference; caught a real `Identity`-node export gotcha from coincidentally-identical untrained LayerNorm weights

## Roadmap

0. Landscape research & gap analysis — done
1. Educational prototype (tiny NN in Python/C/C++) — done
2. Native runtime core (Tensor, Linear, ReLU, Softmax, LayerNorm, GELU) — done ([native/common.hpp](native/common.hpp))
3. Model equivalence (real trained model) — done (exp9: real Indian-name gender classifier, 100% prediction agreement)
4. Benchmarking vs. strong baselines — done, revised toward cold-invocation
5. Packaging — done (safetensors + ONNX + native formats, machine-readable benchmark data)
6. Developer experience (CLI) — done ([native/sajal.cpp](native/sajal.cpp): `inspect`/`run`/`bench`, in-process dispatch across model kinds)
7. Hugging Face integration — done ([sajalmadan09/indian-name-gender-classifier](https://huggingface.co/sajalmadan09/indian-name-gender-classifier), published)
8. Transformer support — done (exp11: real pretrained BERT, [native/bert_model.hpp](native/bert_model.hpp), fp32-tight equivalence on 10 real sentences)
9. Hardware optimization (SIMD/CUDA) — Apple Accelerate/AMX only so far; no CUDA hardware available
10. Research publication
11. Model compiler (ONNX → native C++, not hand-porting) — in progress (exp15-19: [python/sajal_compile.py](python/sajal_compile.py), compiles a complete single-head transformer encoder block, bit-identical to hand-written references throughout; multi-head reshape/transpose not yet supported)
