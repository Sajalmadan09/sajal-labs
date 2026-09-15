"""Experiment 8: does exp7's cold-invocation advantage (15-155x, measured at
exp1/exp2's two fixed configs) actually hold across model size, or was it
an artifact of those two particular sizes? Reuses exp3's sweep artifacts
(artifacts/sweep/d{16,32,64,128,192,256}/) and exp7's methodology (true
external wall-clock per process, interleaved round-robin trials) — the
transformer-only counterpart to exp7, at 6 sizes instead of 1.
"""
import json
import pathlib

from bench_common import percentiles, time_invocation

ROOT = pathlib.Path(__file__).parent.parent
TRIALS = 20
D_MODELS = [16, 32, 64, 128, 192, 256]


def bench_size(d_model):
    artifacts_dir = ROOT / "artifacts" / "sweep" / f"d{d_model}"
    py = str(ROOT / ".venv/bin/python")
    configs = [
        ("native", ["../native/transformer", str(artifacts_dir), "once"]),
        ("pytorch", [py, "bench_pytorch_transformer.py", str(artifacts_dir), "--once"]),
        ("onnx", [py, "bench_onnx_transformer.py", str(artifacts_dir), "--once"]),
    ]

    wall_times = {name: [] for name, _ in configs}
    for _ in range(TRIALS):
        for name, cmd in configs:
            wall_times[name].append(time_invocation(cmd, cwd=ROOT / "python"))

    row = {"d_model": d_model}
    for name, _ in configs:
        p = percentiles(wall_times[name])
        row[f"{name}_p50_ms"] = p["p50_ms"]
        row[f"{name}_p95_ms"] = p["p95_ms"]
    row["native_vs_pytorch"] = row["pytorch_p50_ms"] / row["native_p50_ms"]
    row["native_vs_onnx"] = row["onnx_p50_ms"] / row["native_p50_ms"]
    return row


def main():
    rows = [bench_size(d) for d in D_MODELS]
    for r in rows:
        print(f"d_model={r['d_model']:>4}  native={r['native_p50_ms']:.2f}ms  "
              f"onnx={r['onnx_p50_ms']:.2f}ms  pytorch={r['pytorch_p50_ms']:.2f}ms  "
              f"native_vs_onnx={r['native_vs_onnx']:.1f}x  native_vs_pytorch={r['native_vs_pytorch']:.1f}x")

    out = {"experiment": "exp8-cold-invocation-sweep", "trials": TRIALS, "rows": rows}
    out_dir = ROOT / "research/experiments/exp8-cold-invocation-sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
