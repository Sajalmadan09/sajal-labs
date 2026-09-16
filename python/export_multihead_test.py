"""exp20 test model: multi-head self-attention — exp18's single-head
attention plus the Reshape/Transpose splitting real transformer blocks use
to divide D into (num_heads, d_head), attend per head, then merge back.
This is exp2's TinyTransformerBlock's actual attention shape (4 heads),
the one piece exp19 named as still missing.
"""
import pathlib

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "multihead_test"
D = 16
N_HEADS = 4
D_HEAD = D // N_HEADS
SEQ_LEN = 8


class MultiHeadAttention(nn.Module):
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
        q = self.q(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)  # [heads, seq, d_head]
        k = self.k(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        v = self.v(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        scores = (q @ k.transpose(-2, -1)) * self.scale  # [heads, seq, seq]
        attn = torch.softmax(scores, dim=-1)
        ctx = attn @ v  # [heads, seq, d_head]
        ctx = ctx.transpose(0, 1).reshape(seq, self.d)
        return self.o(ctx)


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = MultiHeadAttention().eval()

    torch.manual_seed(123)
    x = torch.randn(SEQ_LEN, D)  # ONE sequence, not N independent examples
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
    print(f"  multi-head attention, seq_len={SEQ_LEN}, d={D}, n_heads={N_HEADS}")


if __name__ == "__main__":
    main()
