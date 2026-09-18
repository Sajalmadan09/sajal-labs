"""exp28 test model: the decomposed (opset<20) form of GPT-2's actual GELU
activation — transformers.activations.NewGELUActivation, imported and
used directly (not re-implemented), so the exported ONNX graph is
GPT-2's real decomposition, not a guess at what it might look like.

A generic `F.gelu(x, approximate='tanh')` call decomposes DIFFERENTLY at
opset17 (x**3 via two Muls instead of Pow, and the constant multiplies in
a different order) — this is why NewGELUActivation is imported directly
rather than reconstructed: the two are the same math, but not the same
graph, and try_match_tanh_gelu only recognizes the one GPT-2 itself
actually emits (verified against this exact class for that reason).
GPT-2's own real export (opset 20) uses the FUSED Gelu op with
approximate="tanh" instead — this decomposed form is exercised
separately here so both code paths (fused-attribute and decomposed) have
their own regression coverage.
"""
import pathlib

import numpy as np
import torch
import torch.nn as nn
from transformers.activations import NewGELUActivation

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "tanh_gelu_test"
D = 16
D_FF = 32
N_TEST = 20


class M(nn.Module):
    def __init__(self, d=D, d_ff=D_FF):
        super().__init__()
        self.fc1 = nn.Linear(d, d_ff)
        self.act = NewGELUActivation()
        self.fc2 = nn.Linear(d_ff, d)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = M().eval()

    torch.manual_seed(123)
    x = torch.randn(N_TEST, D)
    with torch.no_grad():
        ref = model(x)

    x.numpy().astype(np.float32).tofile(ARTIFACTS / "test_inputs.bin")
    ref.numpy().astype(np.float32).tofile(ARTIFACTS / "ref_outputs.bin")
    (ARTIFACTS / "test_config.txt").write_text(f"{D} {N_TEST}\n")

    torch.onnx.export(
        model, (x,), str(ARTIFACTS / "model.onnx"),
        input_names=["input"], output_names=["output"], opset_version=17, dynamo=False,
    )

    print(f"exported to {ARTIFACTS}")
    print(f"  decomposed tanh-GELU (GPT-2's real NewGELUActivation), d={D}, d_ff={D_FF}")


if __name__ == "__main__":
    main()
