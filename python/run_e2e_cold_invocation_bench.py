"""exp10's benchmark: does paying real feature-extraction cost (not just the
matmul) change the cold-invocation picture from exp7-9? Same methodology
(external wall-clock, interleaved trials, `once` mode) — the only difference
from exp9's cold-invocation numbers is that these three programs now
tokenize/featurize a raw name string every time, instead of using a
precomputed dummy input.
"""
import json
import pathlib

from bench_common import percentiles, time_invocation

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts" / "gender_classifier"
TRIALS = 30


def main():
    py = str(ROOT / ".venv/bin/python")
    configs = [
        ("native_e2e", ["../native/gender_predict", str(ARTIFACTS), "once"]),
        ("pytorch_e2e", [py, "predict_pytorch.py", str(ARTIFACTS), "sajal", "--once"]),
        ("onnx_e2e", [py, "predict_onnx.py", str(ARTIFACTS), "sajal", "--once"]),
    ]

    wall_times = {name: [] for name, _ in configs}
    for _ in range(TRIALS):
        for name, cmd in configs:
            wall_times[name].append(time_invocation(cmd, cwd=ROOT / "python"))

    results = []
    for name, _ in configs:
        result = {"impl": name, "trials": TRIALS}
        result.update(percentiles(wall_times[name]))
        print(f"{name:>12}  p50={result['p50_ms']:.2f}ms  p95={result['p95_ms']:.2f}ms")
        results.append(result)

    out = {"experiment": "exp10-e2e-cold-invocation", "trials": TRIALS, "results": results}
    out_dir = ROOT / "research/experiments/exp10-e2e-native-pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
