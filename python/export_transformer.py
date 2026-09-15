"""Same raw-float32 export pattern as export_model.py (see its docstring for
why no custom format), applied to the transformer block's weight set."""
import pathlib

import numpy as np
import torch

from tiny_transformer import SEQ_LEN, D_MODEL, N_HEADS, D_FF, TinyTransformerBlock

N_TEST = 20
ARTIFACTS = pathlib.Path(__file__).parent.parent / "artifacts" / "transformer"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def main():
    torch.manual_seed(0)
    model = TinyTransformerBlock().eval()

    for name in ["q_proj", "k_proj", "v_proj", "out_proj", "ff1", "ff2"]:
        layer = getattr(model, name)
        save_f32(ARTIFACTS / f"{name}_weight.bin", layer.weight)
        save_f32(ARTIFACTS / f"{name}_bias.bin", layer.bias)
    for name in ["ln1", "ln2"]:
        layer = getattr(model, name)
        save_f32(ARTIFACTS / f"{name}_weight.bin", layer.weight)
        save_f32(ARTIFACTS / f"{name}_bias.bin", layer.bias)
    torch.save(model.state_dict(), ARTIFACTS / "model_state_dict.pt")

    (ARTIFACTS / "shapes.txt").write_text(f"{SEQ_LEN} {D_MODEL} {N_HEADS} {D_FF} {N_TEST}\n")

    torch.manual_seed(123)
    test_inputs = torch.randn(N_TEST, SEQ_LEN, D_MODEL)
    save_f32(ARTIFACTS / "test_inputs.bin", test_inputs)

    with torch.no_grad():
        ref_outputs = torch.stack([model(test_inputs[i]) for i in range(N_TEST)])
    save_f32(ARTIFACTS / "ref_outputs.bin", ref_outputs)

    torch.onnx.export(
        model,
        (test_inputs[0],),
        str(ARTIFACTS / "model.onnx"),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )

    macs = (
        3 * SEQ_LEN * D_MODEL * D_MODEL  # Q,K,V projections
        + 2 * N_HEADS * SEQ_LEN * SEQ_LEN * (D_MODEL // N_HEADS)  # scores + ctx
        + SEQ_LEN * D_MODEL * D_MODEL  # output projection
        + 2 * SEQ_LEN * D_MODEL * D_FF  # FFN
    )
    print(f"exported to {ARTIFACTS}")
    print(f"  seq_len={SEQ_LEN} d_model={D_MODEL} n_heads={N_HEADS} d_ff={D_FF}")
    print(f"  approx MACs/forward pass: {macs:,} ({macs / 9472:.0f}x exp1's tiny MLP)")


if __name__ == "__main__":
    main()
