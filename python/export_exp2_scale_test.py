"""exp22 test model: exp2's ACTUAL TinyTransformerBlock (seq_len=32,
d_model=256, n_heads=4, d_ff=1024) — reused directly (imported, not
re-implemented) so the compiled artifact and native/transformer.cpp (exp2's
original hand-written driver, unchanged since exp2) run the exact same
architecture, letting the compiler's output be benchmarked directly against
exp2's own published baseline instead of only against toy-sized (d=16)
compiler test models.

Writes two parallel sets of artifacts from the SAME weights and input:
  1. model.onnx + test_config.txt/test_inputs.bin/ref_outputs.bin, for
     sajal_compile.py (its usual convention, exp15-21).
  2. shapes.txt + q_proj_weight.bin etc., transformer_model.hpp's exact
     load_transformer() naming convention, so native/transformer.cpp can
     load and run THIS SAME model unmodified.

Same Identity-node export gotcha as exp19/20/21 applies (default-init
LayerNorms export as byte-identical, deduplicated via an unsupported
Identity node) — same fix: perturb ln1/ln2's weights before export. This
does not change exp2's architecture, only which specific (still random)
weight values it starts from.
"""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from tiny_transformer import TinyTransformerBlock, SEQ_LEN, D_MODEL, N_HEADS, D_FF  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "exp2_scale_test"


def save_f32(path, tensor):
    tensor.detach().numpy().astype(np.float32).tofile(path)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = TinyTransformerBlock().eval()
    with torch.no_grad():  # avoid the Identity-aliasing gotcha (see exp19)
        model.ln1.weight.add_(torch.randn(D_MODEL) * 0.01)
        model.ln1.bias.add_(torch.randn(D_MODEL) * 0.01)
        model.ln2.weight.add_(torch.randn(D_MODEL) * 0.01)
        model.ln2.bias.add_(torch.randn(D_MODEL) * 0.01)

    torch.manual_seed(123)
    x = torch.randn(SEQ_LEN, D_MODEL)

    with torch.no_grad():
        ref_output = model(x)

    # (1) sajal_compile.py convention
    save_f32(ARTIFACTS / "test_inputs.bin", x)
    save_f32(ARTIFACTS / "ref_outputs.bin", ref_output)
    (ARTIFACTS / "test_config.txt").write_text(f"{D_MODEL} {SEQ_LEN}\n")
    torch.onnx.export(
        model, (x,), str(ARTIFACTS / "model.onnx"),
        input_names=["input"], output_names=["output"], opset_version=20, dynamo=False,
    )

    # (2) transformer_model.hpp / native/transformer.cpp convention
    (ARTIFACTS / "shapes.txt").write_text(f"{SEQ_LEN} {D_MODEL} {N_HEADS} {D_FF} 1\n")
    save_f32(ARTIFACTS / "q_proj_weight.bin", model.q_proj.weight)
    save_f32(ARTIFACTS / "q_proj_bias.bin", model.q_proj.bias)
    save_f32(ARTIFACTS / "k_proj_weight.bin", model.k_proj.weight)
    save_f32(ARTIFACTS / "k_proj_bias.bin", model.k_proj.bias)
    save_f32(ARTIFACTS / "v_proj_weight.bin", model.v_proj.weight)
    save_f32(ARTIFACTS / "v_proj_bias.bin", model.v_proj.bias)
    save_f32(ARTIFACTS / "out_proj_weight.bin", model.out_proj.weight)
    save_f32(ARTIFACTS / "out_proj_bias.bin", model.out_proj.bias)
    save_f32(ARTIFACTS / "ff1_weight.bin", model.ff1.weight)
    save_f32(ARTIFACTS / "ff1_bias.bin", model.ff1.bias)
    save_f32(ARTIFACTS / "ff2_weight.bin", model.ff2.weight)
    save_f32(ARTIFACTS / "ff2_bias.bin", model.ff2.bias)
    save_f32(ARTIFACTS / "ln1_weight.bin", model.ln1.weight)
    save_f32(ARTIFACTS / "ln1_bias.bin", model.ln1.bias)
    save_f32(ARTIFACTS / "ln2_weight.bin", model.ln2.weight)
    save_f32(ARTIFACTS / "ln2_bias.bin", model.ln2.bias)
    # transformer.cpp's run_mode reads test_inputs.bin too (n_test=1 in
    # shapes.txt above) — same file, same [seq_len,d_model] tensor already
    # written for sajal_compile.py's convention, no separate copy needed.

    print(f"exported to {ARTIFACTS}")
    print(f"  exp2 scale: seq_len={SEQ_LEN}, d_model={D_MODEL}, n_heads={N_HEADS}, d_ff={D_FF}")


if __name__ == "__main__":
    main()
