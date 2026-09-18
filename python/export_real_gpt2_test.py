"""exp28 test model: a real GPT-2 (124M, "gpt2") decoder block — the
compiler's first decoder/causal model, and its first test at hidden=768.

Inspecting GPT-2's actual traced graph first (as always,
/tmp/gpt2layer0.onnx during development) showed several genuinely new
things versus every encoder model tested through exp27:
  - Conv1D (GPT-2's own Linear substitute) explicitly flattens to 2D
    before its matmul and reshapes back after — an extra Reshape pair
    around every Gemm that plain nn.Linear never surfaces. Closed
    directly: try_match_identity_reshape recognizes this as a no-op
    batch=1 squeeze/unsqueeze (this project's convention already treats
    every tensor as [n, dim] with no real batch dim).
  - Conv1D's Gemm has transB=0 (weight stored [in,out], the mirror of the
    usual transB=1/[out,in]) — closed directly, reusing exp26's
    weight_transposed mechanism.
  - The activation is "gelu_new" (tanh approximation), a different
    8-node decomposition than exp16/24's exact/erf formula — closed
    directly via try_match_tanh_gelu.
  - Pre-norm residual placement (LayerNorm BEFORE the sub-block, not
    after) — needed NO changes at all: the DAG-based IR (exp17) walks
    whatever topological order a graph has, it never assumed post-norm.
  - Q/K/V come from ONE combined Conv1D (c_attn) producing 3*d_model,
    split via an ONNX `Split` node — NOT yet supported (would need the
    multihead-attention codegen to address per-QKV-slot offsets into a
    shared buffer, on top of the existing per-head offsets it already
    does; a real, named gap, not attempted this round).
  - GPT-2's own SDPA-derived causal mask is a different decomposition
    than exp27's (`Equal`+additive-bias-`Add` instead of `Cast`+
    select-`Where`) — also not attempted; exp27's own convention is used
    below instead.

So: real weights, real architecture (768 hidden, 12 heads, 3072 FFN, real
pre-norm order, real per-layer LayerNorm eps), but Q/K/V expressed as 3
separate nn.Linear (weights sliced from the real c_attn Conv1D, verified
byte-for-byte faithful — see main()'s equivalence check) and causal
masking expressed via exp27's already-supported `masked_fill` idiom,
rather than tracing GPT2Block.forward() directly. Same strategy exp25
used for BERT, for the same reason: some real gaps are worth closing
directly (this experiment closed 3), and some are worth naming honestly
and routing around rather than expanding scope indefinitely.
"""
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "real_gpt2_test"
MODEL_NAME = "gpt2"
N_LAYERS = 2
SENTENCE = "Sajal Labs is a research project about native machine learning inference."


class RewiredGPT2Block(nn.Module):
    def __init__(self, hf_block, d, d_ff, n_heads, eps):
        super().__init__()
        self.d, self.n_heads, self.d_head = d, n_heads, d // n_heads
        self.ln1 = nn.LayerNorm(d, eps=eps)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d, eps=eps)
        self.ff1 = nn.Linear(d, d_ff)
        self.ff2 = nn.Linear(d_ff, d)
        self.scale = self.d_head ** -0.5
        with torch.no_grad():
            self.ln1.weight.copy_(hf_block.ln_1.weight)
            self.ln1.bias.copy_(hf_block.ln_1.bias)
            self.ln2.weight.copy_(hf_block.ln_2.weight)
            self.ln2.bias.copy_(hf_block.ln_2.bias)
            # Conv1D stores weight [in,out] (the mirror of nn.Linear's
            # [out,in]) — c_attn.weight is [d, 3*d]; slicing it directly
            # (before any transpose) keeps each slice's own [in,out] shape.
            wq, wk, wv = hf_block.attn.c_attn.weight.split(d, dim=1)
            bq, bk, bv = hf_block.attn.c_attn.bias.split(d, dim=0)
            self.q.weight.copy_(wq.T)
            self.q.bias.copy_(bq)
            self.k.weight.copy_(wk.T)
            self.k.bias.copy_(bk)
            self.v.weight.copy_(wv.T)
            self.v.bias.copy_(bv)
            self.o.weight.copy_(hf_block.attn.c_proj.weight.T)
            self.o.bias.copy_(hf_block.attn.c_proj.bias)
            self.ff1.weight.copy_(hf_block.mlp.c_fc.weight.T)
            self.ff1.bias.copy_(hf_block.mlp.c_fc.bias)
            self.ff2.weight.copy_(hf_block.mlp.c_proj.weight.T)
            self.ff2.bias.copy_(hf_block.mlp.c_proj.bias)

    def forward(self, x):  # x: [seq, d] — batch=1 squeezed, this project's convention
        seq = x.shape[0]
        x1 = self.ln1(x)
        q = self.q(x1).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        k = self.k(x1).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        v = self.v(x1).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ctx = attn @ v
        ctx = ctx.transpose(0, 1).reshape(seq, self.d)
        attn_out = self.o(ctx)
        x = x + attn_out
        x2 = self.ln2(x)
        h = self.ff2(F.gelu(self.ff1(x2), approximate="tanh"))
        return x + h


class RewiredGPT2Stack(nn.Module):
    def __init__(self, hf_blocks, d, d_ff, n_heads, eps):
        super().__init__()
        self.blocks = nn.ModuleList([RewiredGPT2Block(b, d, d_ff, n_heads, eps) for b in hf_blocks])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    hf_model = GPT2Model.from_pretrained(MODEL_NAME).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    cfg = hf_model.config

    encoded = tokenizer(SENTENCE, return_tensors="pt")
    with torch.no_grad():
        emb = hf_model.wte(encoded["input_ids"]) + hf_model.wpe(
            torch.arange(encoded["input_ids"].shape[1])
        )
        x_hf = emb
        for layer in hf_model.h[:N_LAYERS]:
            x_hf = layer(x_hf)
        hf_reference = x_hf[0]  # HF's own real forward pass, for the equivalence claim

    x = emb[0]  # [seq, hidden]
    model = RewiredGPT2Stack(hf_model.h[:N_LAYERS], cfg.n_embd, 4 * cfg.n_embd,
                              cfg.n_head, cfg.layer_norm_epsilon).eval()
    with torch.no_grad():
        rewired_output = model(x)

    max_diff_vs_hf = (rewired_output - hf_reference).abs().max().item()
    print(f"rewired-wrapper vs. real GPT-2's own forward pass: max abs diff = {max_diff_vs_hf:.3e}")

    seq_len = x.shape[0]
    save_f32(ARTIFACTS / "test_inputs.bin", x)
    save_f32(ARTIFACTS / "ref_outputs.bin", rewired_output)
    (ARTIFACTS / "test_config.txt").write_text(f"{cfg.n_embd} {seq_len}\n")
    (ARTIFACTS / "equivalence_vs_gpt2.txt").write_text(
        f"sentence: {SENTENCE}\nseq_len: {seq_len}\nmax_abs_diff_vs_real_gpt2_forward: {max_diff_vs_hf}\n"
    )

    torch.onnx.export(
        model, (x,), str(ARTIFACTS / "model.onnx"),
        input_names=["input"], output_names=["output"], opset_version=20, dynamo=False,
    )

    print(f"exported to {ARTIFACTS}")
    print(f"  real {MODEL_NAME}: {N_LAYERS} decoder blocks, hidden={cfg.n_embd}, "
          f"heads={cfg.n_head}, ffn={4 * cfg.n_embd}, seq_len={seq_len}")


if __name__ == "__main__":
    main()
