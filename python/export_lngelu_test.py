"""exp16 test model: Linear -> LayerNorm -> GELU -> Linear -> Softmax. A real
(if synthetic) architecture chosen specifically to exercise LayerNorm and
GELU in isolation from attention's added complexity (residual/skip
connections require DAG support, not just a sequential chain — deliberately
out of scope here, see exp16's results.md). Random weights: this is a
correctness/codegen test, not an accuracy claim, matching exp5/exp12's use
of synthetic weights for mechanism-isolation experiments.

opset_version=20 (not 17, used elsewhere): ONNX only got a single fused
Gelu op at opset 20; at opset 17 PyTorch decomposes GELU into 5 primitive
ops (Div/Erf/Add/Mul/Mul), which would need subgraph pattern-matching this
experiment deliberately doesn't attempt yet.
"""
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "lngelu_test"
IN_DIM, HIDDEN_DIM, OUT_DIM = 32, 64, 10
N_TEST = 20


class LNGeluNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, HIDDEN_DIM)
        self.ln = nn.LayerNorm(HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, OUT_DIM)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = LNGeluNet().eval()

    torch.manual_seed(123)
    test_inputs = torch.randn(N_TEST, IN_DIM)
    save_f32(ARTIFACTS / "test_inputs.bin", test_inputs)
    (ARTIFACTS / "test_config.txt").write_text(f"{IN_DIM} {N_TEST}\n")

    with torch.no_grad():
        ref_outputs = model(test_inputs)
    save_f32(ARTIFACTS / "ref_outputs.bin", ref_outputs)

    torch.onnx.export(
        model, (test_inputs[:1],), str(ARTIFACTS / "model.onnx"),
        input_names=["input"], output_names=["output"], opset_version=20, dynamo=False,
    )

    print(f"exported to {ARTIFACTS}")
    print(f"  Linear({IN_DIM}->{HIDDEN_DIM}) -> LayerNorm -> GELU -> Linear({HIDDEN_DIM}->{OUT_DIM}) -> Softmax")


if __name__ == "__main__":
    main()
