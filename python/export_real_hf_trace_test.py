"""exp26 test model: bert-tiny's real 2-layer encoder, traced directly via
Hugging Face's own BertLayer.forward() (dynamo=False) — NOT rewired into
this project's own attention convention the way exp25's test model was.
Validates the two gaps exp26 closed (MatMul+Add as Linear, 4D-batched
attention with split pre-matmul scaling) against the real thing they were
built for, not just an isolated synthetic case.

The dummy input used for tracing matters: an earlier attempt using random
[1,6,128] noise produced an inconsistent graph (Gemm for one layer,
MatMul+Add for the other, plus an unexplained Gather node) — exporting
with the REAL embedding tensor for a real sentence, as this script does,
produced neither. Not fully root-caused; noted honestly in exp26's
results.md rather than assumed to be understood.
"""
import pathlib

import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "real_hf_2layer_test"
MODEL_NAME = "prajjwal1/bert-tiny"
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
    model = BertModel.from_pretrained(MODEL_NAME).eval()
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    encoded = tokenizer(SENTENCE, return_tensors="pt")
    with torch.no_grad():
        emb = model.embeddings(encoded["input_ids"])  # real embeddings, real sentence
        x = emb
        for layer in model.encoder.layer:
            x = layer(x)
        hf_reference = x[0]

    seq_len = emb.shape[1]
    save_f32(ARTIFACTS / "test_inputs.bin", emb[0])
    save_f32(ARTIFACTS / "ref_outputs.bin", hf_reference)
    (ARTIFACTS / "test_config.txt").write_text(f"{model.config.hidden_size} {seq_len}\n")

    wrapper = LayerStack(model.encoder.layer).eval()
    torch.onnx.export(
        wrapper, (emb,), str(ARTIFACTS / "model.onnx"),
        input_names=["hidden"], output_names=["output"], opset_version=20, dynamo=False,
    )

    print(f"exported to {ARTIFACTS}")
    print(f"  real {MODEL_NAME} encoder, HF's own BertLayer.forward(), no rewiring, seq_len={seq_len}")


if __name__ == "__main__":
    main()
