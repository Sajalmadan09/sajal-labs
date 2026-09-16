"""exp18 test model: single-head self-attention (Q/K/V/output projections,
scaled dot-product attention, softmax over the sequence dimension) — the
smallest real attention variant, deliberately without the multi-head
reshape/transpose machinery exp2/exp11-14's hand-written headers use.
Chosen specifically to isolate the genuinely new capability (MatMul
between two RUNTIME tensors, recognized as a fused pattern by the
compiler) from multi-head splitting, which is a separate, larger step.

Unlike exp15-17's test models, this one's "test_config.txt" n_test doubles
as the sequence length: attention mixes information across all n rows in
one forward() call (they attend to each other), unlike every previous IR
op which treated rows independently. One sequence, not N independent
examples — matching how exp2/exp11-14's real attention was always tested.
"""
import pathlib

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "attention_test"
D = 16
SEQ_LEN = 8


class SingleHeadAttention(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.scale = d ** -0.5

    def forward(self, x):  # x: [seq_len, d]
        Q, K, V = self.q(x), self.k(x), self.v(x)
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        ctx = attn @ V
        return self.o(ctx)


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = SingleHeadAttention().eval()

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
    print(f"  single-head attention, seq_len={SEQ_LEN}, d={D}")


if __name__ == "__main__":
    main()
