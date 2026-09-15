"""Equivalence check: does the C++ forward pass match PyTorch's, and by how
much? Per research/papers.md (Goldberg 1991, Shanmugavelu et al. 2024), exact
bit-for-bit equality is NOT expected — floating-point addition/multiplication
aren't associative, and BLAS accumulates matmul sums in a different order
than PyTorch's own kernels do. What we're checking is that the *tolerance* is
small and explainable, not that it's zero.
"""
import json
import pathlib

import numpy as np

ARTIFACTS = pathlib.Path(__file__).parent.parent / "artifacts"


def main():
    in_dim, hidden_dim, out_dim, n_test = map(
        int, (ARTIFACTS / "shapes.txt").read_text().split()
    )

    ref = np.fromfile(ARTIFACTS / "ref_outputs.bin", dtype=np.float32).reshape(n_test, out_dim)
    native = np.fromfile(ARTIFACTS / "native_outputs.bin", dtype=np.float32).reshape(n_test, out_dim)

    abs_err = np.abs(ref - native)
    max_abs_err = float(abs_err.max())
    mean_abs_err = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean((ref - native) ** 2)))

    cos_sim = float(np.mean([
        np.dot(ref[i], native[i]) / (np.linalg.norm(ref[i]) * np.linalg.norm(native[i]) + 1e-12)
        for i in range(n_test)
    ]))

    ref_pred = ref.argmax(axis=1)
    native_pred = native.argmax(axis=1)
    prediction_agreement = float((ref_pred == native_pred).mean())

    result = {
        "n_test": n_test,
        "max_abs_error": max_abs_err,
        "mean_abs_error": mean_abs_err,
        "rmse": rmse,
        "mean_cosine_similarity": cos_sim,
        "prediction_agreement": prediction_agreement,
        "tolerance_note": (
            "fp32 matmul via BLAS (Accelerate) vs PyTorch's own kernels; "
            "operation-order differences are expected per IEEE-754 non-associativity, "
            "not a bug. See research/papers.md sections 8-9."
        ),
    }

    print(json.dumps(result, indent=2))
    (ARTIFACTS / "equivalence_result.json").write_text(json.dumps(result, indent=2))

    if prediction_agreement < 1.0:
        print("\nWARNING: predicted classes disagree on at least one test vector.")
    if max_abs_err > 1e-4:
        print(f"\nNOTE: max_abs_error {max_abs_err:.2e} exceeds 1e-4 — worth a closer look, "
              f"though softmax outputs in [0,1] with fp32 (~1e-6 to 1e-7 epsilon) commonly "
              f"accumulate error in this range after two matmuls + exp/sum.")


if __name__ == "__main__":
    main()
