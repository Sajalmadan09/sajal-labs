"""Same raw-float32 export pattern as export_model.py (see its docstring for
why no custom format), applied to the transformer block's weight set.
export() is factored out so exp3's sweep can call it per-size instead of
duplicating this file per model width.
"""
import pathlib

import numpy as np
import torch

from tiny_transformer import SEQ_LEN, D_MODEL, N_HEADS, D_FF, TinyTransformerBlock

ROOT = pathlib.Path(__file__).parent.parent


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def export(artifacts_dir, seq_len, d_model, n_heads, d_ff, n_test=20, seed=0):
    artifacts_dir = pathlib.Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    model = TinyTransformerBlock(seq_len, d_model, n_heads, d_ff).eval()

    for name in ["q_proj", "k_proj", "v_proj", "out_proj", "ff1", "ff2"]:
        layer = getattr(model, name)
        save_f32(artifacts_dir / f"{name}_weight.bin", layer.weight)
        save_f32(artifacts_dir / f"{name}_bias.bin", layer.bias)
    for name in ["ln1", "ln2"]:
        layer = getattr(model, name)
        save_f32(artifacts_dir / f"{name}_weight.bin", layer.weight)
        save_f32(artifacts_dir / f"{name}_bias.bin", layer.bias)
    torch.save(model.state_dict(), artifacts_dir / "model_state_dict.pt")

    (artifacts_dir / "shapes.txt").write_text(f"{seq_len} {d_model} {n_heads} {d_ff} {n_test}\n")

    torch.manual_seed(seed + 123)
    test_inputs = torch.randn(n_test, seq_len, d_model)
    save_f32(artifacts_dir / "test_inputs.bin", test_inputs)

    with torch.no_grad():
        ref_outputs = torch.stack([model(test_inputs[i]) for i in range(n_test)])
    save_f32(artifacts_dir / "ref_outputs.bin", ref_outputs)

    torch.onnx.export(
        model,
        (test_inputs[0],),
        str(artifacts_dir / "model.onnx"),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )

    macs = (
        3 * seq_len * d_model * d_model  # Q,K,V projections
        + 2 * n_heads * seq_len * seq_len * (d_model // n_heads)  # scores + ctx
        + seq_len * d_model * d_model  # output projection
        + 2 * seq_len * d_model * d_ff  # FFN
    )
    return macs


def main():
    artifacts_dir = ROOT / "artifacts" / "transformer"
    macs = export(artifacts_dir, SEQ_LEN, D_MODEL, N_HEADS, D_FF)
    print(f"exported to {artifacts_dir}")
    print(f"  seq_len={SEQ_LEN} d_model={D_MODEL} n_heads={N_HEADS} d_ff={D_FF}")
    print(f"  approx MACs/forward pass: {macs:,} ({macs / 9472:.0f}x exp1's tiny MLP)")


if __name__ == "__main__":
    main()
