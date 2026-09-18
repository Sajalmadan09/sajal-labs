"""exp27 test model: causal (autoregressive) multi-head self-attention —
exp20's multi-head attention plus a standard strict upper-triangular mask
(`scores.masked_fill(causal_mask, -inf)`, row i attends to columns <= i
only), the piece needed for decoder/autoregressive models that exp20-26's
encoder-only test models never exercised.

Inspecting the exported graph first (as always) showed the exact pattern:
`Trilu(ones_matrix, k=1) -> Cast -> Where(mask, -inf, scores)` sitting
between the scale step and Softmax — see sajal_compile.py's
try_match_causal_mask docstring.
"""
import pathlib

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "causal_attention_test"
D = 16
N_HEADS = 4
SEQ_LEN = 8


class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d=D, n_heads=N_HEADS):
        super().__init__()
        self.d, self.n_heads, self.d_head = d, n_heads, d // n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.scale = self.d_head ** -0.5

    def forward(self, x):  # x: [seq_len, d]
        seq = x.shape[0]
        q = self.q(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        k = self.k(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        v = self.v(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ctx = attn @ v
        ctx = ctx.transpose(0, 1).reshape(seq, self.d)
        return self.o(ctx)


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = CausalMultiHeadAttention().eval()

    torch.manual_seed(123)
    x = torch.randn(SEQ_LEN, D)
    save_f32(ARTIFACTS / "test_inputs.bin", x)
    (ARTIFACTS / "test_config.txt").write_text(f"{D} {SEQ_LEN}\n")

    with torch.no_grad():
        ref_output = model(x)
    save_f32(ARTIFACTS / "ref_outputs.bin", ref_output)

    torch.onnx.export(
        model, (x,), str(ARTIFACTS / "model.onnx"),
        input_names=["input"], output_names=["output"], opset_version=20, dynamo=False,
    )

    print(f"exported to {ARTIFACTS}")
    print(f"  causal multi-head attention, seq_len={SEQ_LEN}, d={D}, n_heads={N_HEADS}")


if __name__ == "__main__":
    main()
