"""exp17 test model: y = LayerNorm(x + Linear(GELU(Linear(x)))), then a
classification head. This is the residual feed-forward half of a real
transformer block (exp2/exp11-14's architecture), minus attention — chosen
specifically because it requires the input tensor `x` to be consumed
TWICE, non-adjacently (once by the first Linear, again by the final Add),
which a flat sequential chain (exp15/16) has no way to represent but a DAG
does. Random weights: a codegen-correctness test, not an accuracy claim.
"""
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "residual_test"
D, D_FF, OUT_DIM = 16, 64, 10
N_TEST = 20


class ResidualFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff1 = nn.Linear(D, D_FF)
        self.ff2 = nn.Linear(D_FF, D)
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, OUT_DIM)

    def forward(self, x):
        h = self.ff2(F.gelu(self.ff1(x)))
        y = self.ln(x + h)  # <-- x consumed a second time here, non-adjacently
        return torch.softmax(self.head(y), dim=-1)


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = ResidualFFN().eval()

    torch.manual_seed(123)
    test_inputs = torch.randn(N_TEST, D)
    save_f32(ARTIFACTS / "test_inputs.bin", test_inputs)
    (ARTIFACTS / "test_config.txt").write_text(f"{D} {N_TEST}\n")

    with torch.no_grad():
        ref_outputs = model(test_inputs)
    save_f32(ARTIFACTS / "ref_outputs.bin", ref_outputs)

    torch.onnx.export(
        model, (test_inputs[:1],), str(ARTIFACTS / "model.onnx"),
        input_names=["input"], output_names=["output"], opset_version=20, dynamo=False,
    )

    print(f"exported to {ARTIFACTS}")
    print(f"  x -> Linear({D}->{D_FF}) -> GELU -> Linear({D_FF}->{D}) -> (+x) -> LayerNorm -> Linear({D}->{OUT_DIM}) -> Softmax")


if __name__ == "__main__":
    main()
