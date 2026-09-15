"""Equivalence check for compiler-output artifacts (test_config.txt
convention — see sajal_compile.py's docstring for why this differs from
compare.py's shapes.txt convention). Same methodology as compare.py
otherwise; see its docstring for why exact bit-for-bit equality isn't the
bar."""
import json
import pathlib
import sys

import numpy as np

ARTIFACTS = pathlib.Path(sys.argv[1])


def main():
    in_dim, n_test = map(int, (ARTIFACTS / "test_config.txt").read_text().split())
    ref = np.fromfile(ARTIFACTS / "ref_outputs.bin", dtype=np.float32).reshape(n_test, -1)
    native = np.fromfile(ARTIFACTS / "native_outputs.bin", dtype=np.float32).reshape(n_test, -1)
    out_dim = ref.shape[1]

    abs_err = np.abs(ref - native)
    max_abs_err = float(abs_err.max())
    mean_abs_err = float(abs_err.mean())
    rmse = float(np.sqrt(np.mean((ref - native) ** 2)))
    cos_sim = float(np.mean([
        np.dot(ref[i], native[i]) / (np.linalg.norm(ref[i]) * np.linalg.norm(native[i]) + 1e-12)
        for i in range(n_test)
    ]))
    prediction_agreement = float((ref.argmax(axis=1) == native.argmax(axis=1)).mean())

    result = {
        "n_test": n_test, "out_dim": out_dim,
        "max_abs_error": max_abs_err, "mean_abs_error": mean_abs_err, "rmse": rmse,
        "mean_cosine_similarity": cos_sim, "prediction_agreement": prediction_agreement,
    }
    print(json.dumps(result, indent=2))
    (ARTIFACTS / "equivalence_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
