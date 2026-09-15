"""Shared BERT export logic — factored out of bert_export.py so exp12's
synthetic width/depth sweep can reuse the exact same weight/ONNX export path
as exp11's real bert-tiny export. Using one shared exporter for both is the
point: it keeps the toolchain (dynamo ONNX exporter, native .bin format)
identical across the whole ablation, so any latency difference the sweep
finds is attributable to architecture shape, not to export-path differences.
"""
import numpy as np
import torch


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def export_bert_native_weights(model, artifacts_dir):
    """model: a transformers BertModel (real or randomly initialized)."""
    cfg = model.config
    emb = model.embeddings
    save_f32(artifacts_dir / "word_embeddings.bin", emb.word_embeddings.weight)
    save_f32(artifacts_dir / "position_embeddings.bin", emb.position_embeddings.weight)
    save_f32(artifacts_dir / "token_type_embeddings.bin", emb.token_type_embeddings.weight)
    save_f32(artifacts_dir / "emb_ln_weight.bin", emb.LayerNorm.weight)
    save_f32(artifacts_dir / "emb_ln_bias.bin", emb.LayerNorm.bias)

    for i, layer in enumerate(model.encoder.layer):
        prefix = f"layer{i}_"
        attn = layer.attention
        save_f32(artifacts_dir / f"{prefix}q_weight.bin", attn.self.query.weight)
        save_f32(artifacts_dir / f"{prefix}q_bias.bin", attn.self.query.bias)
        save_f32(artifacts_dir / f"{prefix}k_weight.bin", attn.self.key.weight)
        save_f32(artifacts_dir / f"{prefix}k_bias.bin", attn.self.key.bias)
        save_f32(artifacts_dir / f"{prefix}v_weight.bin", attn.self.value.weight)
        save_f32(artifacts_dir / f"{prefix}v_bias.bin", attn.self.value.bias)
        save_f32(artifacts_dir / f"{prefix}attn_out_weight.bin", attn.output.dense.weight)
        save_f32(artifacts_dir / f"{prefix}attn_out_bias.bin", attn.output.dense.bias)
        save_f32(artifacts_dir / f"{prefix}attn_ln_weight.bin", attn.output.LayerNorm.weight)
        save_f32(artifacts_dir / f"{prefix}attn_ln_bias.bin", attn.output.LayerNorm.bias)
        save_f32(artifacts_dir / f"{prefix}ff1_weight.bin", layer.intermediate.dense.weight)
        save_f32(artifacts_dir / f"{prefix}ff1_bias.bin", layer.intermediate.dense.bias)
        save_f32(artifacts_dir / f"{prefix}ff2_weight.bin", layer.output.dense.weight)
        save_f32(artifacts_dir / f"{prefix}ff2_bias.bin", layer.output.dense.bias)
        save_f32(artifacts_dir / f"{prefix}ff_ln_weight.bin", layer.output.LayerNorm.weight)
        save_f32(artifacts_dir / f"{prefix}ff_ln_bias.bin", layer.output.LayerNorm.bias)

    save_f32(artifacts_dir / "pooler_weight.bin", model.pooler.dense.weight)
    save_f32(artifacts_dir / "pooler_bias.bin", model.pooler.dense.bias)

    (artifacts_dir / "config.txt").write_text(
        f"{cfg.num_hidden_layers} {cfg.hidden_size} {cfg.num_attention_heads} "
        f"{cfg.intermediate_size} {cfg.vocab_size} {cfg.max_position_embeddings} {cfg.layer_norm_eps}\n"
    )


def export_bert_onnx(model, artifacts_dir, example_encoded):
    # dynamo=True (the newer torch.export-based exporter) — the legacy
    # TorchScript-based exporter (dynamo=False) fails to trace the installed
    # transformers version's BertModel.forward() (an incompatibility between
    # those two libraries, unrelated to anything in this project's code).
    torch.onnx.export(
        model, (example_encoded["input_ids"], example_encoded["attention_mask"],
                example_encoded["token_type_ids"]),
        str(artifacts_dir / "model.onnx"),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={"input_ids": {1: "seq_len"}, "attention_mask": {1: "seq_len"},
                       "token_type_ids": {1: "seq_len"}, "last_hidden_state": {1: "seq_len"}},
        opset_version=17, dynamo=True,
    )


def export_case(model, artifacts_dir, case_idx, input_ids_1d):
    """input_ids_1d: 1-D torch.LongTensor of token ids (no batch dim)."""
    encoded = {
        "input_ids": input_ids_1d.unsqueeze(0),
        "attention_mask": torch.ones_like(input_ids_1d).unsqueeze(0),
        "token_type_ids": torch.zeros_like(input_ids_1d).unsqueeze(0),
    }
    with torch.no_grad():
        out = model(**encoded)
    input_ids_1d.numpy().astype(np.int32).tofile(artifacts_dir / f"case{case_idx}_input_ids.bin")
    save_f32(artifacts_dir / f"case{case_idx}_ref_hidden.bin", out.last_hidden_state[0])
    save_f32(artifacts_dir / f"case{case_idx}_ref_pooled.bin", out.pooler_output[0])
    return encoded
