"""exp32 test model: real GPT-2's token + position embedding layer
(`wte`, `wpe`), traced directly — the piece that turns raw token ids into
the [seq, hidden] tensor every prior GPT-2 experiment (exp28-31) took as
a given, pre-computed input.

Inspecting the export first (as always) showed why position_ids has to
be a genuine second graph input, not computed internally via
`torch.arange(seq)`: at a FIXED sequence length, PyTorch's exporter
constant-folds that whole computation into a precomputed per-position
bias tensor — there's no Gather for it at all. Forcing seq_len to trace
as dynamic (`dynamic_axes`) recovers the real Gather-based computation
GPT-2's own code performs, at the cost of also making the WHOLE
downstream graph shape-dynamic if this layer were combined with the
transformer stack in one export (2400+ nodes, mostly dynamic-shape
bookkeeping unrelated to embeddings) — which is why this experiment
keeps the embedding layer as its own compiled artifact, chained with
exp31's already-compiled (fixed-shape) 12-layer stack at the artifact
level rather than as one combined ONNX graph. See this experiment's
results.md for the full reasoning and the chained end-to-end check.
"""
import pathlib

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "real_gpt2_embeddings_test"
MODEL_NAME = "gpt2"
SENTENCE = "Sajal Labs is a research project about native machine learning inference."


class Embeddings(nn.Module):
    def __init__(self, wte, wpe):
        super().__init__()
        self.wte = wte
        self.wpe = wpe

    def forward(self, input_ids):  # input_ids: [seq] (1D, unbatched — this project's convention)
        seq = input_ids.shape[0]
        position_ids = torch.arange(seq)
        return self.wte(input_ids) + self.wpe(position_ids)


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model = GPT2Model.from_pretrained(MODEL_NAME).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    encoded = tokenizer(SENTENCE, return_tensors="pt")
    input_ids = encoded["input_ids"][0]

    wrapper = Embeddings(model.wte, model.wpe).eval()
    with torch.no_grad():
        ref_output = wrapper(input_ids)

    seq_len = input_ids.shape[0]
    input_ids.numpy().astype(np.int64).tofile(ARTIFACTS / "test_inputs.bin")
    save_f32(ARTIFACTS / "ref_outputs.bin", ref_output)
    (ARTIFACTS / "test_config.txt").write_text(f"{model.config.n_embd} {seq_len}\n")

    torch.onnx.export(
        wrapper, (input_ids,), str(ARTIFACTS / "model.onnx"),
        input_names=["input_ids"], output_names=["output"],
        dynamic_axes={"input_ids": {0: "seq_len"}, "output": {0: "seq_len"}},
        opset_version=20, dynamo=False,
    )

    print(f"exported to {ARTIFACTS}")
    print(f"  real {MODEL_NAME} embeddings (wte+wpe), seq_len={seq_len}")


if __name__ == "__main__":
    main()
