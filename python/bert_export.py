"""exp11: real pretrained transformer. prajjwal1/bert-tiny (Turc et al.,
"Well-Read Students Learn Better") — a genuine pretrained BERT encoder,
2 layers, hidden=128, 2 heads, intermediate=512, ~4.4M params. Real subword
tokenizer (WordPiece, 30522-entry vocab), real MLM-pretrained weights, not
fine-tuned for a specific downstream task — so this experiment checks
encoder equivalence (last_hidden_state + pooled [CLS] output), the same
"embedding model" category the brief calls out (§6G) alongside classifiers.

Architecture is a real step up from exp2-8's single hand-built block: an
embedding layer (word + position + token_type + LayerNorm) followed by
NUM_LAYERS stacked encoder blocks — native/bert_model.hpp implements this
fresh rather than reusing transformer_model.hpp's TransformerBlock, since
that struct's fixed-size preallocated buffers (exp4's optimization) don't
fit variable-length real sentences, and touching it would risk exp2-8's
benchmarked numbers for no reason.

Export logic itself lives in bert_common.py, shared with exp12's synthetic
width/depth sweep — same exporter for both keeps the toolchain identical
across that ablation.
"""
import pathlib

import numpy as np
import torch
from transformers import BertModel, BertTokenizerFast

from bert_common import export_bert_native_weights, export_bert_onnx, save_f32

MODEL_NAME = "prajjwal1/bert-tiny"
ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "bert_tiny"

# Real sentences, not fabricated gibberish — varying length to exercise
# dynamic sequence-length handling.
TEST_SENTENCES = [
    "Hello.",
    "The quick brown fox jumps over the lazy dog.",
    "Sajal Labs is a research project about native machine learning inference.",
    "Priya trained a small classifier to predict gender from Indian names.",
    "This sentence is intentionally a bit longer than the others to test variable sequence lengths in the native encoder implementation.",
    "Native C++ inference avoids the cost of starting a Python process.",
    "Transformers use self-attention to relate every token to every other token.",
    "BERT was pretrained on masked language modeling and next sentence prediction.",
    "A tiny two-layer encoder is enough to test real equivalence.",
    "Cold invocation latency matters most for serverless and CLI tools.",
]

# exp14: trickier cases specifically stress-testing the WordPiece port —
# punctuation splitting, hyphenation, contractions, out-of-vocabulary words
# forcing multi-piece subword splits, extra whitespace.
TOKENIZER_STRESS_SENTENCES = [
    "don't worry, it's working!",
    "unbelievably-fast native inference.",
    "COVID-19 changed everything in 2020.",
    "  extra   whitespace   everywhere  ",
    "supercalifragilisticexpialidocious",
    "email me at test@example.com, please.",
    "Sajal's C++ port (hopefully) matches exactly.",
]


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model = BertModel.from_pretrained(MODEL_NAME).eval()
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    cfg = model.config

    export_bert_native_weights(model, ARTIFACTS)

    # --- test cases: real sentences, real tokenizer, real reference outputs ---
    case_lengths = []
    for i, text in enumerate(TEST_SENTENCES):
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"][0]
        case_lengths.append(len(input_ids))

        with torch.no_grad():
            out = model(**encoded)

        input_ids.numpy().astype("int32").tofile(ARTIFACTS / f"case{i}_input_ids.bin")
        save_f32(ARTIFACTS / f"case{i}_ref_hidden.bin", out.last_hidden_state[0])
        save_f32(ARTIFACTS / f"case{i}_ref_pooled.bin", out.pooler_output[0])

    (ARTIFACTS / "test_cases.txt").write_text(
        "\n".join(f"{n} {t}" for n, t in zip(case_lengths, TEST_SENTENCES))
    )

    example = tokenizer(TEST_SENTENCES[1], return_tensors="pt")
    export_bert_onnx(model, ARTIFACTS, example)

    # --- exp14: vocab + tokenizer config, for the native WordPiece port ---
    vocab_by_id = [None] * tokenizer.vocab_size
    for tok, idx in tokenizer.get_vocab().items():
        vocab_by_id[idx] = tok
    (ARTIFACTS / "vocab.txt").write_text("\n".join(vocab_by_id))
    (ARTIFACTS / "tokenizer_config.txt").write_text(
        f"{int(tokenizer.do_lower_case)} {tokenizer.cls_token_id} {tokenizer.sep_token_id} "
        f"{tokenizer.unk_token_id} {tokenizer.pad_token_id}\n"
    )

    stress_lengths = []
    for i, text in enumerate(TOKENIZER_STRESS_SENTENCES):
        encoded = tokenizer(text, return_tensors="pt")
        ref_ids = encoded["input_ids"][0]
        stress_lengths.append(len(ref_ids))
        ref_ids.numpy().astype(np.int32).tofile(ARTIFACTS / f"tokcase{i}_ref_ids.bin")
        with torch.no_grad():
            out = model(**encoded)
        save_f32(ARTIFACTS / f"tokcase{i}_ref_hidden.bin", out.last_hidden_state[0])
        save_f32(ARTIFACTS / f"tokcase{i}_ref_pooled.bin", out.pooler_output[0])
    (ARTIFACTS / "tokenizer_test_sentences.txt").write_text(
        "\n".join(f"{n}\t{t}" for n, t in zip(stress_lengths, TOKENIZER_STRESS_SENTENCES))
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"exported {MODEL_NAME} to {ARTIFACTS}")
    print(f"  layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} heads={cfg.num_attention_heads} "
          f"intermediate={cfg.intermediate_size} vocab={cfg.vocab_size}")
    print(f"  params: {n_params:,}  |  {len(TEST_SENTENCES)} test sentences, lengths {case_lengths}")


if __name__ == "__main__":
    main()
