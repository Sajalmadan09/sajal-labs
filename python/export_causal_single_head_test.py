"""exp27 test model: causal (autoregressive) single-head self-attention —
exp18's single-head attention plus a standard strict upper-triangular
mask. Validates try_match_self_attention's causal-mask support
independently of try_match_multihead_attention's (see
export_causal_attention_test.py for the multi-head case) — the two
matchers detect the mask via different code paths (positional vs.
tensor-flow-traced), so both need their own regression coverage.
"""
import pathlib

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "causal_single_head_test"
D = 16
SEQ_LEN = 8


class CausalSingleHeadAttention(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.scale = d ** -0.5

    def forward(self, x):  # x: [seq_len, d]
        seq = x.shape[0]
        Q, K, V = self.q(x), self.k(x), self.v(x)
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ctx = attn @ V
        return self.o(ctx)


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = CausalSingleHeadAttention().eval()

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
    print(f"  causal single-head attention, seq_len={SEQ_LEN}, d={D}")


if __name__ == "__main__":
    main()
