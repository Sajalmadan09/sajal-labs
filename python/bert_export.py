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
"""
import pathlib

import numpy as np
import torch
from transformers import BertModel, BertTokenizerFast

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


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model = BertModel.from_pretrained(MODEL_NAME).eval()
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    cfg = model.config

    # --- weights ---
    emb = model.embeddings
    save_f32(ARTIFACTS / "word_embeddings.bin", emb.word_embeddings.weight)
    save_f32(ARTIFACTS / "position_embeddings.bin", emb.position_embeddings.weight)
    save_f32(ARTIFACTS / "token_type_embeddings.bin", emb.token_type_embeddings.weight)
    save_f32(ARTIFACTS / "emb_ln_weight.bin", emb.LayerNorm.weight)
    save_f32(ARTIFACTS / "emb_ln_bias.bin", emb.LayerNorm.bias)

    for i, layer in enumerate(model.encoder.layer):
        prefix = f"layer{i}_"
        attn = layer.attention
        save_f32(ARTIFACTS / f"{prefix}q_weight.bin", attn.self.query.weight)
        save_f32(ARTIFACTS / f"{prefix}q_bias.bin", attn.self.query.bias)
        save_f32(ARTIFACTS / f"{prefix}k_weight.bin", attn.self.key.weight)
        save_f32(ARTIFACTS / f"{prefix}k_bias.bin", attn.self.key.bias)
        save_f32(ARTIFACTS / f"{prefix}v_weight.bin", attn.self.value.weight)
        save_f32(ARTIFACTS / f"{prefix}v_bias.bin", attn.self.value.bias)
        save_f32(ARTIFACTS / f"{prefix}attn_out_weight.bin", attn.output.dense.weight)
        save_f32(ARTIFACTS / f"{prefix}attn_out_bias.bin", attn.output.dense.bias)
        save_f32(ARTIFACTS / f"{prefix}attn_ln_weight.bin", attn.output.LayerNorm.weight)
        save_f32(ARTIFACTS / f"{prefix}attn_ln_bias.bin", attn.output.LayerNorm.bias)
        save_f32(ARTIFACTS / f"{prefix}ff1_weight.bin", layer.intermediate.dense.weight)
        save_f32(ARTIFACTS / f"{prefix}ff1_bias.bin", layer.intermediate.dense.bias)
        save_f32(ARTIFACTS / f"{prefix}ff2_weight.bin", layer.output.dense.weight)
        save_f32(ARTIFACTS / f"{prefix}ff2_bias.bin", layer.output.dense.bias)
        save_f32(ARTIFACTS / f"{prefix}ff_ln_weight.bin", layer.output.LayerNorm.weight)
        save_f32(ARTIFACTS / f"{prefix}ff_ln_bias.bin", layer.output.LayerNorm.bias)

    save_f32(ARTIFACTS / "pooler_weight.bin", model.pooler.dense.weight)
    save_f32(ARTIFACTS / "pooler_bias.bin", model.pooler.dense.bias)

    (ARTIFACTS / "config.txt").write_text(
        f"{cfg.num_hidden_layers} {cfg.hidden_size} {cfg.num_attention_heads} "
        f"{cfg.intermediate_size} {cfg.vocab_size} {cfg.max_position_embeddings} {cfg.layer_norm_eps}\n"
    )

    # --- test cases: real sentences, real tokenizer, real reference outputs ---
    case_lengths = []
    for i, text in enumerate(TEST_SENTENCES):
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"][0]
        seq_len = len(input_ids)
        case_lengths.append(seq_len)

        with torch.no_grad():
            out = model(**encoded)

        input_ids.numpy().astype(np.int32).tofile(ARTIFACTS / f"case{i}_input_ids.bin")
        save_f32(ARTIFACTS / f"case{i}_ref_hidden.bin", out.last_hidden_state[0])
        save_f32(ARTIFACTS / f"case{i}_ref_pooled.bin", out.pooler_output[0])

    (ARTIFACTS / "test_cases.txt").write_text(
        "\n".join(f"{n} {t}" for n, t in zip(case_lengths, TEST_SENTENCES))
    )

    # ONNX export, using case1's sequence length as the fixed example shape.
    # dynamo=True (the newer torch.export-based exporter) — the legacy
    # TorchScript-based exporter (dynamo=False) fails to trace the installed
    # transformers version's BertModel.forward() (an incompatibility between
    # those two libraries, unrelated to anything in this project's code).
    example = tokenizer(TEST_SENTENCES[1], return_tensors="pt")
    torch.onnx.export(
        model, (example["input_ids"], example["attention_mask"], example["token_type_ids"]),
        str(ARTIFACTS / "model.onnx"),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={"input_ids": {1: "seq_len"}, "attention_mask": {1: "seq_len"},
                       "token_type_ids": {1: "seq_len"}, "last_hidden_state": {1: "seq_len"}},
        opset_version=17, dynamo=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"exported {MODEL_NAME} to {ARTIFACTS}")
    print(f"  layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} heads={cfg.num_attention_heads} "
          f"intermediate={cfg.intermediate_size} vocab={cfg.vocab_size}")
    print(f"  params: {n_params:,}  |  {len(TEST_SENTENCES)} test sentences, lengths {case_lengths}")


if __name__ == "__main__":
    main()
