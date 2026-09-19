"""exp31 test model: real GPT-2's FULL 12-layer decoder stack, traced
directly (no rewiring), same approach as exp30's
export_real_gpt2_direct_test.py but with N_LAYERS=12 instead of 2 — the
model's actual complete depth, not a truncated sample of it.

exp23 found that stacking synthetic blocks exposed a real bug (a global
shape-scan matcher merging branches from different attention instances)
that no single-instance test could have caught. This asks the same
question of exp28-30's real-GPT-2 line: does real depth (not just two
layers, picked mainly to prove "more than one works" at all) surface
anything exp23's fix didn't already cover, now that every individual
per-layer capability (combined QKV via Split, the additive causal mask,
tanh-GELU, Conv1D's transB=0/reshape-wrapping) is closed?
"""
import pathlib

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "real_gpt2_full_stack_test"
MODEL_NAME = "gpt2"
SENTENCE = "Sajal Labs is a research project about native machine learning inference."


class LayerStack(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model = GPT2Model.from_pretrained(MODEL_NAME).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    n_layers = model.config.n_layer

    encoded = tokenizer(SENTENCE, return_tensors="pt")
    with torch.no_grad():
        emb = model.wte(encoded["input_ids"]) + model.wpe(torch.arange(encoded["input_ids"].shape[1]))
        x_hf = emb
        for layer in model.h:  # ALL layers — the model's real full depth
            x_hf = layer(x_hf)
        hf_reference = x_hf[0]

    seq_len = emb.shape[1]
    save_f32(ARTIFACTS / "test_inputs.bin", emb[0])
    save_f32(ARTIFACTS / "ref_outputs.bin", hf_reference)
    (ARTIFACTS / "test_config.txt").write_text(f"{model.config.n_embd} {seq_len}\n")

    wrapper = LayerStack(model.h).eval()
    torch.onnx.export(
        wrapper, (emb,), str(ARTIFACTS / "model.onnx"),
        input_names=["hidden"], output_names=["output"], opset_version=20, dynamo=False,
    )

    print(f"exported to {ARTIFACTS}")
    print(f"  real {MODEL_NAME}: ALL {n_layers} decoder blocks (full depth), traced directly, seq_len={seq_len}")


if __name__ == "__main__":
    main()
