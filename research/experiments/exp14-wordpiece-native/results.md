# Experiment 14 — WordPiece Ported to C++: A Fully End-to-End Real Transformer

**Why this experiment**: exp11 validated the BERT encoder itself but fed it precomputed token IDs — a real deployment gets raw text. This ports BERT's actual WordPiece tokenizer to C++ (mirroring exp10's approach for the gender classifier's simpler bigram features, but now for a real ~30K-entry subword vocabulary and a genuinely non-trivial algorithm), completing the fully-native pipeline: raw string in, embedding out, zero Python anywhere.

## The port

[native/wordpiece.hpp](../../../native/wordpiece.hpp) implements BERT's actual two-stage tokenizer: `BasicTokenizer` (lowercase, whitespace/punctuation splitting) followed by `WordpieceTokenizer` (greedy longest-match-first over the vocabulary, `##`-prefixed continuation pieces, whole-word `[UNK]` on failure) — the real algorithm, not an approximation. Stated scope limitation, upfront rather than discovered later: this is an ASCII-focused port. Real BERT's tokenizer does full Unicode category lookups (accent stripping, CJK character isolation); this implementation guards `<cctype>` calls to the ASCII range (bytes 0-127) and passes higher bytes through as word characters. Verified exactly correct on this project's real English test sentences, not claimed correct for arbitrary Unicode input.

## Equivalence: exact token match, not just close

Two levels checked, both on 7 genuinely tricky stress-test sentences (contractions, hyphens, an out-of-vocabulary word forcing an 11-piece subword split, an email address, multiple whitespace) plus the original 10 sentences from exp11:

**Tokenization — exact match required, not a tolerance.** Every one of 17 test sentences produced byte-for-byte identical token IDs to HF's `BertTokenizerFast`, including the hard cases:

| Sentence | Result |
|---|---|
| `"don't worry, it's working!"` | exact match (apostrophe/contraction splitting) |
| `"unbelievably-fast native inference."` | exact match (hyphen splitting) |
| `"COVID-19 changed everything in 2020."` | exact match |
| `"  extra   whitespace   everywhere  "` | exact match |
| `"supercalifragilisticexpialidocious"` | exact match (11-piece subword split, out-of-vocabulary word) |
| `"email me at test@example.com, please."` | exact match |
| `"Sajal's C++ port (hopefully) matches exactly."` | exact match |

**Full pipeline — raw text through tokenizer AND encoder, fp32-tight**: max hidden-state error 4.05e-06, max pooled-output error 8.02e-06 across all 7 stress cases — consistent with exp11's precomputed-token numbers (9.5e-06/2.2e-06), confirming the tokenizer isn't just producing plausible-looking IDs but IDs that drive the encoder to the same real numbers HF produces.

## Benchmark: the tokenizer library choice matters as much as the inference engine

True end-to-end cold invocation (raw text in, prediction out, external wall-clock, 30 trials). Discovered mid-benchmark and worth reporting as found, not smoothed over: the first ONNX Runtime script used `transformers.BertTokenizerFast` for preprocessing (convenient, reuses exp11's code) and its cold-start cost exploded — because ONNX Runtime not needing Python for *inference* doesn't mean the *tokenizer* is free, and `transformers` is a heavy import regardless of which inference engine consumes its output. Checked whether a leaner tokenizer library changes that: HuggingFace's standalone `tokenizers` package (Rust-based, no `transformers` dependency) produces byte-identical token IDs at a fraction of the import cost — so both are reported, not just the more flattering one.

| Implementation | Cold invocation p50 | vs. native |
|---|---:|---:|
| **Native C++** | 11.37ms | — |
| ONNX Runtime + `tokenizers` (lean) | 95.72ms | 8.4x slower |
| ONNX Runtime + `transformers` (convenient) | 2,495.61ms | 219.5x slower |
| PyTorch + `transformers` | 4,973.91ms | 437.5x slower |

Full data: [cold_invocation_results.json](cold_invocation_results.json).

**The tokenizer library choice alone is a 26x swing** (2,495.61ms → 95.72ms, same ONNX Runtime, same model, only the preprocessing library changed). A "fast inference engine" claim is meaningless if the preprocessing step in front of it isn't held to the same standard — and getting that right requires knowing to reach for `tokenizers` instead of the more obvious/convenient `transformers` import, a choice most people wouldn't think to make without measuring.

## Interpretation

This is the strongest version yet of the project's central finding, for two reasons. First, it holds even against the *best-case* Python deployment: native beats even the properly-optimized ONNX+lean-tokenizer combination by 8.4x, not just the convenient-but-heavy one. Second, it surfaces a finding bigger than this project's own scope: **cold-invocation performance is a property of the whole dependency chain, not just the inference engine** — Sajal Labs' native pipeline sidesteps this entire class of problem by construction (there's no separate "pick the right preprocessing library" decision to get right or wrong), while a Python deployment's actual cold-start number depends on library choices that have nothing to do with model architecture or inference framework.

## Caveats

ASCII-only tokenizer scope, stated above — a model needing real Unicode/CJK support would need that work done first. Single fixed benchmark sentence (matching exp11's `case1`). The `tokenizers`-vs-`transformers` finding is specific to this task (WordPiece via a standalone library existing) — not every tokenizer scheme has as lightweight a standalone implementation available.

## Next experiment

The equivalence/benchmarking methodology has now been validated end-to-end on a real model, real preprocessing, and real (occasionally tricky) input at every stage this project's roadmap called for. Per research/gap-analysis.md, remaining directions are less about proving the core hypothesis further and more about breadth: package this BERT model for Hugging Face (mirroring the gender classifier's Phase 7 work), extend `sajal`'s CLI to detect and drive this "wordpiece-mlp" kind alongside the existing three, or shift toward quantization now that full-precision behavior is this thoroughly characterized.
