"""exp29 test model: multi-head attention with a COMBINED Q/K/V
projection — one Linear producing 3*d outputs, split via ONNX Split into
Q/K/V — instead of three dedicated Linears (exp20's convention). This is
GPT-2's real `c_attn` shape (exp28 named this gap and deliberately routed
around it rather than closing it); scoped here without exp28's other
real-model complications (no causal masking, no Conv1D, no tanh-GELU) so
the Split mechanism itself gets clean, isolated regression coverage.
"""
import pathlib

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "combined_qkv_test"
D = 16
N_HEADS = 4
SEQ_LEN = 8


class CombinedQKVAttention(nn.Module):
    def __init__(self, d=D, n_heads=N_HEADS):
        super().__init__()
        self.d, self.n_heads, self.d_head = d, n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.scale = self.d_head ** -0.5

    def forward(self, x):  # x: [seq_len, d]
        seq = x.shape[0]
        q, k, v = self.qkv(x).split(self.d, dim=-1)
        q = q.view(seq, self.n_heads, self.d_head).transpose(0, 1)
        k = k.view(seq, self.n_heads, self.d_head).transpose(0, 1)
        v = v.view(seq, self.n_heads, self.d_head).transpose(0, 1)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        ctx = attn @ v
        ctx = ctx.transpose(0, 1).reshape(seq, self.d)
        return self.o(ctx)


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = CombinedQKVAttention().eval()

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
    print(f"  combined-QKV multi-head attention, seq_len={SEQ_LEN}, d={D}, n_heads={N_HEADS}")


if __name__ == "__main__":
    main()
