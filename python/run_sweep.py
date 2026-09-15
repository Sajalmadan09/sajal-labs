"""Experiment 3: sweep d_model to locate the crossover between exp1's
dispatch-bound regime (native ~6-13x faster, warm latency) and exp2's
compute-bound regime (~1x, statistical tie) — see
research/experiments/exp2-tiny-transformer/results.md §6.

Same architecture as exp2 (TinyTransformerBlock), same seq_len=32 and
n_heads=4, only d_model varies; d_ff scales with it at the standard 4x
ratio so "widening the model" scales every matmul dimension together
instead of introducing a second free variable.
"""
import json
import pathlib
import subprocess
import sys

from bench_common import run_timed, dir_size_mb
from export_transformer import export

ROOT = pathlib.Path(__file__).parent.parent
SEQ_LEN, N_HEADS = 32, 4
D_MODELS = [16, 32, 64, 128, 192, 256]
EXPERIMENT_NAME = sys.argv[1] if len(sys.argv) > 1 else "exp3-size-sweep"


def run_one(d_model):
    d_ff = 4 * d_model
    artifacts_dir = ROOT / "artifacts" / "sweep" / f"d{d_model}"
    macs = export(artifacts_dir, SEQ_LEN, d_model, N_HEADS, d_ff)

    # Equivalence check, same as exp1/exp2: native `run` mode, then compare.py.
    subprocess.run(
        ["../native/transformer", str(artifacts_dir), "run"],
        cwd=ROOT / "python", check=True, capture_output=True,
    )
    equiv = json.loads(
        subprocess.run(
            ["../.venv/bin/python", "compare_transformer.py", str(artifacts_dir)],
            cwd=ROOT / "python", check=True, capture_output=True, text=True,
        ).stdout
    )

    native = run_timed(["../native/transformer", str(artifacts_dir), "bench"], cwd=ROOT / "python")
    pytorch = run_timed(["../.venv/bin/python", "bench_pytorch_transformer.py", str(artifacts_dir)],
                         cwd=ROOT / "python")
    onnx = run_timed(["../.venv/bin/python", "bench_onnx_transformer.py", str(artifacts_dir)],
                      cwd=ROOT / "python")

    return {
        "d_model": d_model, "d_ff": d_ff, "macs": macs,
        "max_abs_error": equiv["max_abs_error"],
        "native_p50_ms": native["p50_ms"], "native_peak_rss_mb": native["peak_rss_mb"],
        "pytorch_p50_ms": pytorch["p50_ms"], "onnx_p50_ms": onnx["p50_ms"],
        "native_vs_pytorch": pytorch["p50_ms"] / native["p50_ms"],
        "native_vs_onnx": onnx["p50_ms"] / native["p50_ms"],
    }


def main():
    rows = [run_one(d) for d in D_MODELS]
    for r in rows:
        print(f"d_model={r['d_model']:>4}  macs={r['macs']:>10,}  "
              f"native={r['native_p50_ms']:.5f}ms  onnx={r['onnx_p50_ms']:.5f}ms  "
              f"native_vs_onnx={r['native_vs_onnx']:.2f}x")

    out = {"experiment": EXPERIMENT_NAME, "seq_len": SEQ_LEN, "n_heads": N_HEADS, "rows": rows}
    out_dir = ROOT / "research/experiments" / EXPERIMENT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
