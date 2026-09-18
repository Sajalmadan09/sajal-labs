"""exp30 test model: real GPT-2's actual decoder block trace, traced
DIRECTLY via model.h[i](x) — no rewiring at all, unlike exp28's
export_real_gpt2_test.py, which deliberately routed around this model's
combined Q/K/V projection and causal-mask decomposition by rebuilding the
block with separate nn.Linears and exp27's masked_fill idiom.

exp29 closed the combined-QKV gap; this closes the other one exp28 named
and deferred — GPT-2's own causal mask, which decomposes very differently
from exp27's synthetic models:
  - the all-ones matrix comes from `Expand(1.0, dynamic_shape)`, not a
    literal Constant (the shape is computed at runtime from Q/K's own
    shapes via Shape/Slice/Concat)
  - `Trilu`'s `upper` attribute is explicitly 0 (lower triangle), not the
    ONNX-spec default of 1 — found by extracting this node's actual
    runtime output via onnxruntime and reading it directly, after an
    initial by-hand derivation (assuming upper defaulted to 1, since the
    attribute wasn't printed) gave the OPPOSITE, wrong polarity
  - the boolean mask comes from `Equal(trilu_out, 0.0)` (inverted — true
    where Trilu is 0), not a `Cast` (direct pass-through)
  - the mask is combined via `Where(mask,-inf,0.0) -> Add(scores, ·)`
    (an additive bias computed separately), not `Where(mask,-inf,scores)`
    directly
Two more real quirks surfaced getting this to actually compile, both
found by trying and reading the exact failure, not anticipated: K's own
head-split Reshape has TWO Transpose consumers in this real export (the
real K^T used in the attention matmul, plus a "throwaway" transpose that
only feeds the mask's dynamic-shape computation) — try_match_multihead_
attention's branch resolution had always assumed exactly one; and the
dead Shape/Slice/Concat/Expand subgraph computing that dynamic shape had
to be explicitly marked as consumed (harmless, orphaned bookkeeping this
compiler never needs at runtime since `n` is already a parameter) rather
than left to be rejected as an unrecognized standalone op.
"""
import pathlib

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "real_gpt2_direct_test"
MODEL_NAME = "gpt2"
N_LAYERS = 2
SENTENCE = "Sajal Labs is a research project about native machine learning inference."


class LayerStack(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model = GPT2Model.from_pretrained(MODEL_NAME).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    encoded = tokenizer(SENTENCE, return_tensors="pt")
    with torch.no_grad():
        emb = model.wte(encoded["input_ids"]) + model.wpe(torch.arange(encoded["input_ids"].shape[1]))
        x_hf = emb
        for layer in model.h[:N_LAYERS]:
            x_hf = layer(x_hf)  # GPT2Block.forward returns a plain Tensor, not a tuple
        hf_reference = x_hf[0]

    seq_len = emb.shape[1]
    save_f32(ARTIFACTS / "test_inputs.bin", emb[0])
    save_f32(ARTIFACTS / "ref_outputs.bin", hf_reference)
    (ARTIFACTS / "test_config.txt").write_text(f"{model.config.n_embd} {seq_len}\n")

    wrapper = LayerStack(model.h[:N_LAYERS]).eval()
    torch.onnx.export(
        wrapper, (emb,), str(ARTIFACTS / "model.onnx"),
        input_names=["hidden"], output_names=["output"], opset_version=20, dynamo=False,
    )

    print(f"exported to {ARTIFACTS}")
    print(f"  real {MODEL_NAME}: {N_LAYERS} decoder blocks, TRACED DIRECTLY (no rewiring), seq_len={seq_len}")


if __name__ == "__main__":
    main()
