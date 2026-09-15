"""exp14: does the native WordPiece port produce EXACTLY the same token IDs
as HF's real tokenizer? Unlike numerical equivalence (fp32 tolerance),
tokenization is deterministic — this is an exact-match bar, not a
tolerance one."""
import pathlib

import numpy as np

ARTIFACTS = pathlib.Path(__file__).parent.parent / "artifacts" / "bert_tiny"


def main():
    lines = (ARTIFACTS / "tokenizer_test_sentences.txt").read_text().splitlines()
    cases = [(int(n), t) for n, t in (line.split("\t", 1) for line in lines)]
    native_lines = (ARTIFACTS / "tokenizer_check_native.txt").read_text().splitlines()

    all_match = True
    max_hidden_err, max_pooled_err = 0.0, 0.0
    for i, (n, text) in enumerate(cases):
        ref_ids = np.fromfile(ARTIFACTS / f"tokcase{i}_ref_ids.bin", dtype=np.int32).tolist()
        native_ids = [int(x) for x in native_lines[i].split()]
        ids_match = ref_ids == native_ids
        all_match &= ids_match

        ref_h = np.fromfile(ARTIFACTS / f"tokcase{i}_ref_hidden.bin", dtype=np.float32).reshape(n, 128)
        nat_h = np.fromfile(ARTIFACTS / f"tokcase{i}_native_hidden.bin", dtype=np.float32).reshape(n, 128)
        ref_p = np.fromfile(ARTIFACTS / f"tokcase{i}_ref_pooled.bin", dtype=np.float32)
        nat_p = np.fromfile(ARTIFACTS / f"tokcase{i}_native_pooled.bin", dtype=np.float32)
        h_err = float(np.abs(ref_h - nat_h).max())
        p_err = float(np.abs(ref_p - nat_p).max())
        max_hidden_err = max(max_hidden_err, h_err)
        max_pooled_err = max(max_pooled_err, p_err)

        status = "OK" if ids_match else "TOKEN MISMATCH"
        print(f"[{status}] \"{text}\"  hidden_err={h_err:.2e}  pooled_err={p_err:.2e}")

    print(f"\ntokens: {'ALL MATCH EXACTLY' if all_match else 'SOME MISMATCHED'}")
    print(f"end-to-end (raw text -> embedding): max_hidden_err={max_hidden_err:.2e}  max_pooled_err={max_pooled_err:.2e}")
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
