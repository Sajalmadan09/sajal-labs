import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ_LEN, D_MODEL, N_HEADS, D_FF = 32, 256, 4, 1024


class TinyTransformerBlock(nn.Module):
    """One pre-norm-free (post-norm, like the original Transformer) encoder
    block: multi-head self-attention + residual + LayerNorm, then a
    GELU feed-forward + residual + LayerNorm. Hand-rolled rather than
    nn.MultiheadAttention so both the math and the native C++ port are
    explicit instead of hidden behind a fused kernel.

    Shape is a constructor arg (not a module-level constant) so exp3 can
    sweep d_model without redefining this class per size.
    """

    def __init__(self, seq_len=SEQ_LEN, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF):
        super().__init__()
        self.seq_len, self.d_model, self.n_heads, self.d_ff = seq_len, d_model, n_heads, d_ff
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):  # x: [seq, d_model], batch size 1 with the batch dim squeezed out
        seq = x.shape[0]
        q = self.q_proj(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)  # [heads,seq,d_head]
        k = self.k_proj(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        v = self.v_proj(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [heads, seq, seq]
        attn = torch.softmax(scores, dim=-1)
        ctx = attn @ v  # [heads, seq, d_head]
        ctx = ctx.transpose(0, 1).reshape(seq, self.d_model)  # back to [seq, d_model]

        x = self.ln1(x + self.out_proj(ctx))  # residual + LayerNorm
        ff = self.ff2(F.gelu(self.ff1(x)))
        x = self.ln2(x + ff)
        return x
