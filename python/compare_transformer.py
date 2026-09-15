"""Same equivalence methodology as compare.py (see its docstring), adapted
for the transformer block's [n_test, seq_len, d_model] output shape. No
'prediction' to agree on here (this block isn't a classifier), so per-element
error stats are the primary signal, plus per-token cosine similarity."""
import json
import pathlib

import numpy as np

ARTIFACTS = pathlib.Path(__file__).parent.parent / "artifacts" / "transformer"


def main():
    seq_len, d_model, n_heads, d_ff, n_test = map(
        int, (ARTIFACTS / "shapes.txt").read_text().split()
    )

    ref = np.fromfile(ARTIFACTS / "ref_outputs.bin", dtype=np.float32).reshape(n_test, seq_len, d_model)
    native = np.fromfile(ARTIFACTS / "native_outputs.bin", dtype=np.float32).reshape(
        n_test, seq_len, d_model
    )

    abs_err = np.abs(ref - native)
    max_abs_err = float(abs_err.max())
    mean_abs_err = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean((ref - native) ** 2)))

    ref_flat = ref.reshape(-1, d_model)
    native_flat = native.reshape(-1, d_model)
    dots = np.sum(ref_flat * native_flat, axis=1)
    norms = np.linalg.norm(ref_flat, axis=1) * np.linalg.norm(native_flat, axis=1) + 1e-12
    cos_sim = float(np.mean(dots / norms))

    result = {
        "n_test": n_test,
        "shape": [seq_len, d_model],
        "max_abs_error": max_abs_err,
        "mean_abs_error": mean_abs_err,
        "rmse": rmse,
        "mean_cosine_similarity_per_token": cos_sim,
        "tolerance_note": (
            "6 matmuls + softmax + 2 LayerNorms + GELU, fp32, ~2700x more MACs than exp1's MLP — "
            "expect larger accumulated error than exp1, still fp32-scale, not a bug. "
            "See research/papers.md sections 8-9."
        ),
    }

    print(json.dumps(result, indent=2))
    (ARTIFACTS / "equivalence_result.json").write_text(json.dumps(result, indent=2))

    if max_abs_err > 1e-2:
        print(f"\nNOTE: max_abs_error {max_abs_err:.2e} is notably larger than exp1's ~1e-8 — "
              f"worth checking whether it's proportional to the extra matmul/LayerNorm depth "
              f"or signals an actual bug (e.g. a LayerNorm eps or GELU formula mismatch).")


if __name__ == "__main__":
    main()
