"""exp19 test model: one complete single-head transformer encoder block —
y1 = LayerNorm(x + Attention(x)); y2 = LayerNorm(y1 + FFN(y1)) — combining
exp17's residual Add with exp18's fused self-attention. This is exp2's
real TinyTransformerBlock architecture (post-norm, same residual+LayerNorm
placement), minus multi-head splitting (still not supported — see exp18).

A real gotcha, found and worked around rather than silently avoided: with
default (untrained) nn.LayerNorm init, ln1 and ln2 start with byte-identical
weight/bias values (weight=1, bias=0), and PyTorch's ONNX exporter
deduplicates them via Identity nodes aliasing one LayerNorm's params to the
other's — an op this compiler doesn't support, unrelated to anything about
attention or residual connections. Fixed by perturbing ln1/ln2's weights
to different random values, same as any other layer's random init.
"""
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "full_block_test"
D, D_FF = 16, 64
SEQ_LEN = 8


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d=D, d_ff=D_FF):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, d_ff)
        self.ff2 = nn.Linear(d_ff, d)
        self.ln2 = nn.LayerNorm(d)
        self.scale = d ** -0.5
        with torch.no_grad():  # avoid the Identity-aliasing gotcha described above
            self.ln1.weight.add_(torch.randn(d) * 0.01)
            self.ln1.bias.add_(torch.randn(d) * 0.01)
            self.ln2.weight.add_(torch.randn(d) * 0.01)
            self.ln2.bias.add_(torch.randn(d) * 0.01)

    def forward(self, x):
        Q, K, V = self.q(x), self.k(x), self.v(x)
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        ctx = attn @ V
        attn_out = self.o(ctx)
        x1 = self.ln1(x + attn_out)  # residual 1
        h = self.ff2(F.gelu(self.ff1(x1)))
        x2 = self.ln2(x1 + h)  # residual 2
        return x2


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = TransformerEncoderBlock().eval()

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
    print(f"  one full single-head transformer encoder block, seq_len={SEQ_LEN}, d={D}, d_ff={D_FF}")


if __name__ == "__main__":
    main()
