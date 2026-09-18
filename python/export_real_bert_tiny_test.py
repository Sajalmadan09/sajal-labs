"""exp25: the compiler against a REAL pretrained model — prajjwal1/bert-tiny
(2 layers, hidden=128, 2 heads, intermediate=512, real MLM-trained weights),
not a synthetic random-weight test model.

Inspecting the graph first (as always) showed why this can't just export
model.encoder.layer directly and compile it: tracing HF's actual forward
pass (torch.onnx.export, dynamo=False, default SDPA attention) produces —
- Linear as MatMul+Add, not Gemm (a new op-pair this compiler doesn't
  recognize as "linear" at all)
- 4D attention (explicit batch dim: reshape to [1,seq,H,Dh], perm
  [0,2,1,3]/[0,2,3,1], not this compiler's 3D [seq,H,Dh]/perm [1,0,2])
- split sqrt-scaling (Q and K each multiplied by d_head**-0.25 BEFORE the
  matmul, not one scale after it — a different pattern than exp20/22's)
- an unexplained Gather node
- inconsistent Gemm-vs-MatMul+Add across layers in the SAME graph when
  multiple layers are traced together (a PyTorch export quirk, not
  anything in this project's code)
This is exactly why exp11 hand-ported BERT to C++ rather than compiling it
in the first place — HF's own traced graph is a moving target across
opsets/versions/attention backends, unrelated to whether the MATH the
compiler already supports is correct.

The honest, well-scoped test: rewire the REAL pretrained weights into this
project's own attention convention (single combined scale, 3D reshape,
batch-squeezed [seq,d] input — identical to exp20/21/23's synthetic test
models), verified to match HF's actual forward pass to fp32 tolerance
(2.44e-06 max abs diff on a real sentence — see this experiment's
results.md), then export THAT. Same real weights, same real architecture,
a different (compiler-friendly) but verified-equivalent expression of the
same computation — not a different model.
"""
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "real_bert_tiny_test"
MODEL_NAME = "prajjwal1/bert-tiny"
SENTENCE = "Sajal Labs is a research project about native machine learning inference."


class RewiredBertLayer(nn.Module):
    def __init__(self, hf_layer, d, d_ff, n_heads, eps):
        super().__init__()
        self.d, self.n_heads, self.d_head = d, n_heads, d // n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d, eps=eps)
        self.ff1 = nn.Linear(d, d_ff)
        self.ff2 = nn.Linear(d_ff, d)
        self.ln2 = nn.LayerNorm(d, eps=eps)
        self.scale = self.d_head ** -0.5
        with torch.no_grad():
            self.q.weight.copy_(hf_layer.attention.self.query.weight)
            self.q.bias.copy_(hf_layer.attention.self.query.bias)
            self.k.weight.copy_(hf_layer.attention.self.key.weight)
            self.k.bias.copy_(hf_layer.attention.self.key.bias)
            self.v.weight.copy_(hf_layer.attention.self.value.weight)
            self.v.bias.copy_(hf_layer.attention.self.value.bias)
            self.o.weight.copy_(hf_layer.attention.output.dense.weight)
            self.o.bias.copy_(hf_layer.attention.output.dense.bias)
            self.ln1.weight.copy_(hf_layer.attention.output.LayerNorm.weight)
            self.ln1.bias.copy_(hf_layer.attention.output.LayerNorm.bias)
            self.ff1.weight.copy_(hf_layer.intermediate.dense.weight)
            self.ff1.bias.copy_(hf_layer.intermediate.dense.bias)
            self.ff2.weight.copy_(hf_layer.output.dense.weight)
            self.ff2.bias.copy_(hf_layer.output.dense.bias)
            self.ln2.weight.copy_(hf_layer.output.LayerNorm.weight)
            self.ln2.bias.copy_(hf_layer.output.LayerNorm.bias)

    def forward(self, x):
        seq = x.shape[0]
        q = self.q(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        k = self.k(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        v = self.v(x).view(seq, self.n_heads, self.d_head).transpose(0, 1)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        ctx = attn @ v
        ctx = ctx.transpose(0, 1).reshape(seq, self.d)
        x1 = self.ln1(x + self.o(ctx))
        h = self.ff2(F.gelu(self.ff1(x1)))
        return self.ln2(x1 + h)


class RewiredBertEncoder(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        cfg = hf_model.config
        self.layers = nn.ModuleList([
            RewiredBertLayer(layer, cfg.hidden_size, cfg.intermediate_size,
                              cfg.num_attention_heads, cfg.layer_norm_eps)
            for layer in hf_model.encoder.layer
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    hf_model = BertModel.from_pretrained(MODEL_NAME).eval()
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    cfg = hf_model.config

    encoded = tokenizer(SENTENCE, return_tensors="pt")
    with torch.no_grad():
        emb = hf_model.embeddings(encoded["input_ids"])  # real embeddings, real sentence
        x_batched = emb
        for hf_layer in hf_model.encoder.layer:
            x_batched = hf_layer(x_batched)
        hf_reference = x_batched[0]  # HF's own real forward pass, for the equivalence claim

    x = emb[0]  # [seq, hidden] — batch-squeezed, this project's convention
    model = RewiredBertEncoder(hf_model).eval()
    with torch.no_grad():
        rewired_output = model(x)

    max_diff_vs_hf = (rewired_output - hf_reference).abs().max().item()
    print(f"rewired-wrapper vs. HF's own real forward pass: max abs diff = {max_diff_vs_hf:.3e}")

    seq_len = x.shape[0]
    save_f32(ARTIFACTS / "test_inputs.bin", x)
    save_f32(ARTIFACTS / "ref_outputs.bin", rewired_output)
    (ARTIFACTS / "test_config.txt").write_text(f"{cfg.hidden_size} {seq_len}\n")
    (ARTIFACTS / "equivalence_vs_hf.txt").write_text(
        f"sentence: {SENTENCE}\nseq_len: {seq_len}\nmax_abs_diff_vs_real_hf_forward: {max_diff_vs_hf}\n"
    )

    torch.onnx.export(
        model, (x,), str(ARTIFACTS / "model.onnx"),
        input_names=["input"], output_names=["output"], opset_version=20, dynamo=False,
    )

    print(f"exported to {ARTIFACTS}")
    print(f"  real {MODEL_NAME}: layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
          f"heads={cfg.num_attention_heads} intermediate={cfg.intermediate_size}, seq_len={seq_len}")


if __name__ == "__main__":
    main()
