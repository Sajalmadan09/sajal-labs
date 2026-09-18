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
20. [Multi-head attention](research/experiments/exp20-multihead-attention/results.md) — the last piece from exp2's real architecture (4 heads): a new tensor-flow-based (not positional) pattern matcher finds the non-contiguous `Reshape`/`Transpose` head-split PyTorch's exporter emits, reusing the exact per-head BLAS-slicing trick already validated in `transformer_model.hpp`; bit-identical to a hand-written reference; all six prior compiler targets still pass regression
21. [Multi-head attention + full block](research/experiments/exp21-multihead-full-block/results.md) — combined exp19's residual+FFN block with exp20's multi-head split into exp2's actual `TransformerBlock` architecture in full (4 heads); zero compiler changes needed, bit-identical to a hand-written reference, all seven prior compiler targets still pass regression
22. [Compiler vs. exp2's baseline](research/experiments/exp22-compiler-vs-baseline/results.md) — benchmarked the compiled artifact against exp2's actual hand-written binary, same architecture/weights/methodology: cold-invocation, warm p50, peak RSS, and binary size all statistically indistinguishable; found and fixed a real gap along the way (exp2's own model uses `Div` for attention scaling, which the compiler hadn't recognized — every synthetic test model up to this point happened to use `Mul` instead)
23. [Stacked encoder blocks](research/experiments/exp23-stacked-blocks/results.md) — found a real bug the moment a graph contained more than one attention instance: exp20's matcher found its Q/K/V branches via a global shape scan, which silently merged branches from different blocks; fixed by rewriting the matcher to be purely tensor-flow-traced (never "find anything shaped like X anywhere in the graph"); 2- and 3-block stacks now compile correctly, bit-identical to a hand-written reference, all 9 prior compiler targets still pass regression
24. [Decomposed GELU (opset<20)](research/experiments/exp24-decomposed-gelu/results.md) — widened GELU recognition to the 5-node form ONNX exporters emit before opset 20 (`Div->Erf->Add->Mul->Mul`); a simple positional lookahead sufficed (no branching, unlike attention), verified on both an isolated case and exp21's full multi-head block re-exported at opset 17 with zero other changes needed — bit-identical to the existing opset-20 reference, all 10 prior compiler targets still pass regression
25. [A real Hugging Face model](research/experiments/exp25-real-huggingface-model/results.md) — tried the compiler against prajjwal1/bert-tiny (exp11's model). Tracing HF's own forward pass directly exposed 4 real mismatches (MatMul+Add instead of Gemm, 4D batched attention, split sqrt-scaling, an export-quirk Gather) — too much new work for one experiment. Instead: rewired the real pretrained weights into the compiler's known attention convention, verified numerically faithful to HF's actual output (2.44e-06 max diff), then compiled with **zero compiler changes** — bit-identical to a hand-written reference
26. [MatMul+Add and 4D attention](research/experiments/exp26-matmul-add-4d-attention/results.md) — closed two of exp25's four gaps directly: `MatMul(x,W)->Add(·,bias)` now recognized as Linear (W stored transposed, corrected at weight-extraction time), and the multi-head matcher generalized to 4D-batched shapes and HF's split pre-matmul scaling. Result: bert-tiny's real 2-layer encoder, traced by Hugging Face's own code with **zero rewiring**, compiles and matches HF's actual output to 2.86e-06 — the first compiler target that isn't bit-identical to a hand-written reference, for a fully understood, benign reason (combining two independently-rounded scale factors — real floating-point non-associativity, not a bug)
27. [Causal attention masking](research/experiments/exp27-causal-attention-masking/results.md) — recognizes PyTorch's standard `Trilu->Cast->Where` decomposition of `scores.masked_fill(causal_mask, -inf)` in both attention matchers, needed for autoregressive/decoder models; a modifier on the existing attention ops rather than a new one, bit-identical to hand-written references for both single- and multi-head cases, all 14 prior compiler targets still pass regression
28. [A real GPT-2 decoder](research/experiments/exp28-real-gpt2-decoder/results.md) — a real `gpt2` decoder block surfaced 6 differences from every encoder model tried before it. Closed 3 directly (Conv1D's flatten/unflatten reshape as a no-op, its `transB=0` weight orientation, and `"gelu_new"` as a genuinely separate op from exact GELU — not a rename); 2 turned out to need only a one-line dimension-bookkeeping fix once pre-norm ordering exposed it; 1 (combined Q/K/V via a single `Split`-based projection) named and routed around rather than closed, same call exp25 made for BERT's stray `Gather`. 2 real decoder blocks compile bit-identical to a hand-written reference, all 16 prior compiler targets still pass regression
29. [Combined Q/K/V via Split](research/experiments/exp29-combined-qkv-split/results.md) — closed exp28's deferred gap directly: Q/K/V from one combined projection, addressed as column-slices of a shared buffer via the same BLAS pointer-offset trick already used for per-head slicing, one level up. Took two real bug fixes to get there: the main loop needed an explicit lookahead since `Split` always precedes the node that triggers the match consuming it, and two producer-lookup helpers had silently assumed every node has exactly one output since exp20 — true for 8 experiments, false the instant a real multi-output node showed up. Bit-identical to a hand-written reference, all 18 prior compiler targets still pass regression; real GPT-2's fully-unrewired compile still needs a separate, still-open mask-variant gap closed too
30. [GPT-2's additive causal mask](research/experiments/exp30-additive-causal-mask/results.md) — closed the mask gap exp28 named and exp29 confirmed was still blocking: GPT-2's real mask decomposes as `Equal`+`Where`+`Add` with `Trilu`'s `upper` attribute explicitly 0, not exp27's `Cast`+`Where` with the default. A first by-hand derivation of the polarity was silently backwards — settled by an empirical PyTorch comparison and extracting the actual runtime `Trilu` output via onnxruntime, not further guessing. Getting the real trace to actually compile after that surfaced two more real bugs (K's Reshape has two Transpose consumers in this export; the mask's dead shape-bookkeeping subgraph needed explicit marking). **Real GPT-2, traced directly by Hugging Face's own code with zero rewiring, now compiles** — closing the loop opened at exp25. All 19 prior compiler targets still pass regression

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
11. Model compiler (ONNX → native C++, not hand-porting) — in progress (exp15-30: [python/sajal_compile.py](python/sajal_compile.py), compiles exp2's actual `TransformerBlock` architecture, real bert-tiny, and real GPT-2 (both encoder and decoder, both traced directly from Hugging Face's own unmodified export, zero rewiring); still limited to a single block/layer-stack, not a full model with embeddings/heads, and only the specific ONNX shapes actually observed in real exports so far)
