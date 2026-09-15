"""exp11 equivalence check: real BERT, real sentences, varying lengths.
Same methodology as compare.py/compare_transformer.py — see their
docstrings for why exact bit-for-bit equality isn't expected."""
import json
import pathlib

import numpy as np

ARTIFACTS = pathlib.Path(__file__).parent.parent / "artifacts" / "bert_tiny"


def main():
    lines = (ARTIFACTS / "test_cases.txt").read_text().splitlines()
    cases = [(int(line.split(" ", 1)[0]), line.split(" ", 1)[1]) for line in lines]

    per_case = []
    hidden_max_errs, pooled_max_errs, hidden_cos_sims, pooled_cos_sims = [], [], [], []

    for i, (seq_len, text) in enumerate(cases):
        ref_h = np.fromfile(ARTIFACTS / f"case{i}_ref_hidden.bin", dtype=np.float32).reshape(seq_len, 128)
        nat_h = np.fromfile(ARTIFACTS / f"case{i}_native_hidden.bin", dtype=np.float32).reshape(seq_len, 128)
        ref_p = np.fromfile(ARTIFACTS / f"case{i}_ref_pooled.bin", dtype=np.float32)
        nat_p = np.fromfile(ARTIFACTS / f"case{i}_native_pooled.bin", dtype=np.float32)

        h_err = float(np.abs(ref_h - nat_h).max())
        p_err = float(np.abs(ref_p - nat_p).max())
        h_cos = float(np.mean([
            np.dot(ref_h[j], nat_h[j]) / (np.linalg.norm(ref_h[j]) * np.linalg.norm(nat_h[j]) + 1e-12)
            for j in range(seq_len)
        ]))
        p_cos = float(np.dot(ref_p, nat_p) / (np.linalg.norm(ref_p) * np.linalg.norm(nat_p) + 1e-12))

        hidden_max_errs.append(h_err)
        pooled_max_errs.append(p_err)
        hidden_cos_sims.append(h_cos)
        pooled_cos_sims.append(p_cos)
        per_case.append({"case": i, "seq_len": seq_len, "text": text,
                          "hidden_max_abs_err": h_err, "pooled_max_abs_err": p_err,
                          "hidden_cos_sim": h_cos, "pooled_cos_sim": p_cos})

    result = {
        "n_cases": len(cases),
        "hidden_max_abs_error_overall": max(hidden_max_errs),
        "pooled_max_abs_error_overall": max(pooled_max_errs),
        "mean_hidden_cos_sim": float(np.mean(hidden_cos_sims)),
        "mean_pooled_cos_sim": float(np.mean(pooled_cos_sims)),
        "per_case": per_case,
        "tolerance_note": (
            "fp32 matmul via BLAS (Accelerate) vs PyTorch's own kernels, 2-layer real BERT encoder; "
            "operation-order differences are expected per IEEE-754 non-associativity, not a bug. "
            "See research/papers.md sections 8-9."
        ),
    }

    print(json.dumps({k: v for k, v in result.items() if k != "per_case"}, indent=2))
    (ARTIFACTS / "equivalence_result.json").write_text(json.dumps(result, indent=2))

    print("\nper-case max abs error (hidden):")
    for c in per_case:
        print(f"  case{c['case']} (len={c['seq_len']:>3}): {c['hidden_max_abs_err']:.2e}  \"{c['text'][:50]}\"")


if __name__ == "__main__":
    main()
