# Experiment 11 — A Real Pretrained Transformer

**Why this experiment**: exp2-8's transformer work used a hand-built, randomly-initialized block — real architecture, but not a real pretrained model. Per the project's roadmap (Phase 8) and the original brief's Model 3 ("take a real small pretrained transformer... embedding model... do not immediately use a large LLM"), this ports an actual pretrained model: [`prajjwal1/bert-tiny`](https://huggingface.co/prajjwal1/bert-tiny) (Turc et al., "Well-Read Students Learn Better") — 2 layers, hidden=128, 2 heads, intermediate=512, **4,385,920 real pretrained parameters**, real 30,522-entry WordPiece vocabulary.

## What's genuinely new here vs. exp2-8

- **A real embedding layer**: word + position + token-type embeddings, summed, then LayerNorm — an operator (`Embedding`) the brief's §3 calls out that nothing in this project had implemented before.
- **Multiple stacked encoder layers** (2, not exp2-8's 1), each with real pretrained weights, not random init.
- **A real tokenizer** (WordPiece) producing genuine variable-length token sequences from real English sentences — 10 test sentences, 4 to 25 tokens long.
- **[`native/bert_model.hpp`](../../../native/bert_model.hpp) is a fresh implementation**, not a reuse of `transformer_model.hpp`'s `TransformerBlock` — that struct's fixed-size preallocated scratch buffers (exp4's optimization) are wrong for variable-length real sentences, and touching it would risk exp2-8's already-benchmarked numbers for no reason. The per-layer attention/FFN math is structurally identical (same post-norm residual+LayerNorm pattern, same BLAS per-head stride trick), reimplemented with per-call dynamic sizing instead — correctness for real input first, not yet re-optimized (see Caveats).

## Equivalence: real model, real sentences, still fp32-tight

10 real sentences (lengths 4-25 tokens), each run through both PyTorch's `BertModel` and the native port, checking both the full `last_hidden_state` and the pooled `[CLS]` embedding:

| Metric | Value |
|---|---:|
| Max absolute error (hidden states, all cases) | 9.54e-06 |
| Max absolute error (pooled output, all cases) | 2.19e-06 |
| Mean cosine similarity (hidden states) | 1.0 |
| Mean cosine similarity (pooled) | 1.0000000358 |

Slightly larger than exp2's synthetic single-block error (1.67e-06) but still tight fp32-scale — expected, since this model has 2 stacked layers of real (not random) weights and a genuine embedding lookup adding another op to the chain. No case, at any length, showed a discontinuity or outlier — full per-case breakdown in [results.json-equivalent](../../../artifacts/bert_tiny/equivalence_result.json) (gitignored, regenerable via `python/compare_bert.py`).

## Cold invocation: the largest gap measured in this project so far

| Implementation | Cold invocation p50 | vs. native |
|---|---:|---:|
| **Native C++** | 8.75ms | — |
| ONNX Runtime (CPU) | 87.31ms | 9.98x slower |
| PyTorch + 🤗 transformers | 5,095.88ms | **582x slower** |

The `transformers` library's own import cost dwarfs everything measured before it in this project — plain `torch` alone cost ~400-700ms to import in exp1-10; `transformers` on top of it costs **~5 seconds**. This isn't a criticism of `transformers` (it's a large, general library doing a lot at import time — tokenizers, model registries, framework detection) — it's a direct, honest measurement of what "real deployment stack" actually costs for cold/single-shot invocation, and it's the single largest cold-start number in ten-plus experiments.

## Warm loop: a more nuanced result than exp2-8's collapse

| Implementation | Warm p50 | vs. native |
|---|---:|---:|
| **Native C++** | 0.1115ms | — |
| ONNX Runtime (CPU) | 0.1396ms | 1.25x slower |
| PyTorch + transformers | 0.2748ms | 2.46x slower |

Unlike exp2-6's transformer (which collapsed to ~1.0x parity with ONNX Runtime at `d_model=256`, `head_dim=64`), this real model — same `head_dim=64` as exp2's synthetic block, but narrower overall (`hidden=128` vs `256`) and with 2 stacked layers instead of 1 — retains a modest, real edge (1.25x-2.46x) in the warm loop too. This doesn't fit the simplest version of exp5/exp9's "operator count fully explains it" story cleanly: this model has *more* total BLAS calls than exp2's single block (roughly 2x, one set per layer), yet still shows a native edge exp2 didn't. Plausible explanations not yet isolated: ONNX Runtime may optimize a real, deeper multi-layer graph differently than a single synthetic block; the narrower `hidden=128` dimension may sit in a different part of the dispatch-overhead-vs-compute tradeoff than exp2's `256`. Reported honestly as an open nuance, not forced into the existing narrative.

## Interpretation

The headline claim from exp7-10 — cold-invocation advantage is large, real, and consistent regardless of architecture — holds again here, at the largest scale tested yet (4.4M real params vs. exp9's 22K), and the gap is *larger*, not smaller, once a real library stack (`transformers`, not just `torch`) is the honest comparison. The warm-loop result adds a genuine open question rather than a clean confirmation, which is itself useful: the "operator count, not language" explanation from exp5 is not yet a complete theory, and this real model is evidence for that gap, not just another data point that fits neatly.

## Caveats

- Buffers are dynamically allocated per `forward()` call (correctness-first, matching exp1-4's *original* un-optimized state before exp4's buffer-reuse fix) — not yet re-applying that optimization here. Given exp4 found buffer reuse barely mattered for the synthetic transformer, this is a plausible but unverified assumption for this real model too.
- Single representative sentence (case 1, 12 tokens) used for the cold-invocation and warm-loop benchmarks, for comparability with a fixed input size — the 10-sentence set only exercises the equivalence check, not the benchmark.
- No task head (bert-tiny is a base pretrained encoder, not fine-tuned) — this validates encoder/embedding equivalence, not a downstream classification/task accuracy claim, unlike exp9's gender classifier.
- Tokenization (real WordPiece) is still done in Python, feeding precomputed token IDs to the native side — matching exp1-9's methodology before exp10 went fully end-to-end for the simpler bigram case. WordPiece is meaningfully more complex to port (a ~30K-entry vocabulary and a greedy longest-match algorithm) than character bigrams.

## Next experiment

Two natural directions, not mutually exclusive: (1) port WordPiece tokenization to C++ (mirroring exp10's fully-end-to-end pipeline, now for a real subword tokenizer instead of character bigrams) so this becomes a true "raw text in, embedding out" native pipeline; (2) investigate the warm-loop nuance directly — benchmark exp2's exact synthetic architecture at `hidden=128` (matching bert-tiny's width) to see whether the retained edge here is about width, depth, or being a real (vs. synthetic) graph that ONNX Runtime optimizes differently.
