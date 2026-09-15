import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ_LEN, D_MODEL, N_HEADS, D_FF = 32, 256, 4, 1024
D_HEAD = D_MODEL // N_HEADS


class TinyTransformerBlock(nn.Module):
    """One pre-norm-free (post-norm, like the original Transformer) encoder
    block: multi-head self-attention + residual + LayerNorm, then a
    GELU feed-forward + residual + LayerNorm. Hand-rolled rather than
    nn.MultiheadAttention so both the math and the native C++ port are
    explicit instead of hidden behind a fused kernel.
    """

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL)
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ff1 = nn.Linear(D_MODEL, D_FF)
        self.ff2 = nn.Linear(D_FF, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)

    def forward(self, x):  # x: [seq, d_model], batch size 1 with the batch dim squeezed out
        seq = x.shape[0]
        q = self.q_proj(x).view(seq, N_HEADS, D_HEAD).transpose(0, 1)  # [heads, seq, d_head]
        k = self.k_proj(x).view(seq, N_HEADS, D_HEAD).transpose(0, 1)
        v = self.v_proj(x).view(seq, N_HEADS, D_HEAD).transpose(0, 1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D_HEAD)  # [heads, seq, seq]
        attn = torch.softmax(scores, dim=-1)
        ctx = attn @ v  # [heads, seq, d_head]
        ctx = ctx.transpose(0, 1).reshape(seq, D_MODEL)  # back to [seq, d_model]

        x = self.ln1(x + self.out_proj(ctx))  # residual + LayerNorm
        ff = self.ff2(F.gelu(self.ff1(x)))
        x = self.ln2(x + ff)
        return x
