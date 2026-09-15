"""exp9: the real-model case study. Runs the exact same cold-invocation
methodology as exp7/exp8 (external wall-clock, interleaved trials, `once`
mode) against the trained Indian-name gender classifier instead of a
synthetic model — the point being to confirm exp7/exp8's findings transfer
to something someone would actually deploy, not just hand-built teaching
architectures.
"""
import json
import pathlib

from bench_common import percentiles, time_invocation, run_timed, dir_size_mb

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "gender_classifier"
TRIALS = 30


def main():
    py = str(ROOT / ".venv/bin/python")
    configs = [
        ("native", ["../native/mlp", str(ARTIFACTS), "once"]),
        ("pytorch", [py, "bench_pytorch.py", str(ARTIFACTS), "--once"]),
        ("onnx", [py, "bench_onnx.py", str(ARTIFACTS), "--once"]),
    ]

    wall_times = {name: [] for name, _ in configs}
    for _ in range(TRIALS):
        for name, cmd in configs:
            wall_times[name].append(time_invocation(cmd, cwd=ROOT / "python"))

    cold_invocation = []
    for name, _ in configs:
        result = {"impl": name, "trials": TRIALS}
        result.update(percentiles(wall_times[name]))
        print(f"{name:>10}  p50={result['p50_ms']:.2f}ms  p95={result['p95_ms']:.2f}ms")
        cold_invocation.append(result)

    # Also grab warm-loop latency + RSS, same as exp1-6's methodology, for a full comparison.
    warm = [
        run_timed(["../native/mlp", str(ARTIFACTS), "bench"], cwd=ROOT / "python"),
        run_timed(["../.venv/bin/python", "bench_pytorch.py", str(ARTIFACTS)], cwd=ROOT / "python"),
        run_timed(["../.venv/bin/python", "bench_onnx.py", str(ARTIFACTS)], cwd=ROOT / "python"),
    ]

    training_metrics = json.loads((ARTIFACTS / "training_metrics.json").read_text())
    equivalence = json.loads((ARTIFACTS / "equivalence_result.json").read_text())

    out = {
        "experiment": "exp9-gender-classifier",
        "training_metrics": training_metrics,
        "equivalence": equivalence,
        "cold_invocation": cold_invocation,
        "warm_loop": warm,
        "native_binary_mb": (ROOT / "native" / "mlp").stat().st_size / 1e6,
    }
    out_dir = ROOT / "research/experiments/exp9-gender-classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
