"""Build the reference PyTorch model, export weights as raw float32 + test
vectors + reference outputs, so the C++ side has something to load and
something to check itself against.

No custom binary format here (per the brief's §16 — don't invent a format
before understanding why existing ones exist): each tensor is just its raw
float32 bytes, row-major, written with numpy's .tofile(). Shapes go in a
plain text file since the C++ side has no JSON parser and doesn't need one.
"""
import pathlib

import numpy as np
import torch

from tiny_mlp import IN_DIM, HIDDEN_DIM, OUT_DIM, TinyMLP

N_TEST = 20
ARTIFACTS = pathlib.Path(__file__).parent.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def main():
    torch.manual_seed(0)
    model = TinyMLP().eval()

    save_f32(ARTIFACTS / "fc1_weight.bin", model.fc1.weight)  # [HIDDEN, IN]
    save_f32(ARTIFACTS / "fc1_bias.bin", model.fc1.bias)  # [HIDDEN]
    save_f32(ARTIFACTS / "fc2_weight.bin", model.fc2.weight)  # [OUT, HIDDEN]
    save_f32(ARTIFACTS / "fc2_bias.bin", model.fc2.bias)  # [OUT]
    torch.save(model.state_dict(), ARTIFACTS / "model_state_dict.pt")

    (ARTIFACTS / "shapes.txt").write_text(f"{IN_DIM} {HIDDEN_DIM} {OUT_DIM} {N_TEST}\n")

    torch.manual_seed(123)
    test_inputs = torch.randn(N_TEST, IN_DIM)
    save_f32(ARTIFACTS / "test_inputs.bin", test_inputs)

    with torch.no_grad():
        ref_outputs = model(test_inputs)
    save_f32(ARTIFACTS / "ref_outputs.bin", ref_outputs)

    torch.onnx.export(
        model,
        (test_inputs[:1],),
        str(ARTIFACTS / "model.onnx"),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )

    print(f"exported to {ARTIFACTS}")
    print(f"  fc1: {tuple(model.fc1.weight.shape)}  fc2: {tuple(model.fc2.weight.shape)}")
    print(f"  test_inputs: {tuple(test_inputs.shape)}  ref_outputs: {tuple(ref_outputs.shape)}")


if __name__ == "__main__":
    main()
