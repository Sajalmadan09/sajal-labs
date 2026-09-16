"""exp23 test model: N stacked transformer encoder blocks (multi-head
attention + residual + FFN, exp21's shape) fed into each other — does the
compiler handle a graph containing several independent attention
instances, not just one? Inspecting the exported graph first (as always)
showed PyTorch scopes each block's tensor names distinctly
(`blocks.0.q.weight` vs `blocks.1.q.weight`, etc.) — no naming collision —
but exp20's original multi-head matcher still broke on this, because it
found the right pattern by searching the WHOLE graph for reshapes sharing
one (num_heads, d_head) shape, which merges branches from different blocks
that happen to share that shape. Fixed by making the matcher purely
tensor-flow-traced (see try_match_multihead_attention's docstring).
"""
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "stacked_blocks_test"
D, D_FF = 16, 64
N_HEADS = 4
SEQ_LEN = 8
N_BLOCKS = 2


class MultiHeadTransformerEncoderBlock(nn.Module):
    def __init__(self, d=D, d_ff=D_FF, n_heads=N_HEADS):
        super().__init__()
        self.d, self.n_heads, self.d_head = d, n_heads, d // n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, d_ff)
        self.ff2 = nn.Linear(d_ff, d)
        self.ln2 = nn.LayerNorm(d)
        self.scale = self.d_head ** -0.5
        with torch.no_grad():  # avoid the Identity-aliasing gotcha (see exp19)
            self.ln1.weight.add_(torch.randn(d) * 0.01)
            self.ln1.bias.add_(torch.randn(d) * 0.01)
            self.ln2.weight.add_(torch.randn(d) * 0.01)
            self.ln2.bias.add_(torch.randn(d) * 0.01)

    def forward(self, x):
        seq = x.shape[0]
        q = self.q(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        k = self.k(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        v = self.v(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        ctx = attn @ v
        ctx = ctx.transpose(0, 1).reshape(seq, self.d)
        attn_out = self.o(ctx)
        x1 = self.ln1(x + attn_out)
        h = self.ff2(F.gelu(self.ff1(x1)))
        return self.ln2(x1 + h)


class StackedTransformerEncoder(nn.Module):
    def __init__(self, n_blocks=N_BLOCKS, d=D, d_ff=D_FF, n_heads=N_HEADS):
        super().__init__()
        self.blocks = nn.ModuleList([MultiHeadTransformerEncoderBlock(d, d_ff, n_heads) for _ in range(n_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = StackedTransformerEncoder().eval()

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
    print(f"  {N_BLOCKS} stacked transformer encoder blocks, seq_len={SEQ_LEN}, d={D}, n_heads={N_HEADS}")


if __name__ == "__main__":
    main()
