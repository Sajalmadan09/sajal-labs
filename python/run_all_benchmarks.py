"""Orchestrates all three benchmarks under a common wrapper (/usr/bin/time -l,
for peak RSS on macOS), and reports dependency/binary footprint alongside
latency — per the brief's §12 fairness rules: same machine, same batch size
(1), single-threaded, warmup+iters documented, hardware/compiler recorded
separately in research/environment.md.
"""
import json
import pathlib

from bench_common import run_timed, dir_size_mb

ROOT = pathlib.Path(__file__).parent.parent
ARTIFACTS = ROOT / "artifacts"


def main():
    results = [
        run_timed(["../native/mlp", "../artifacts", "bench"], cwd=ROOT / "python"),
        run_timed(["../.venv/bin/python", "bench_pytorch.py"], cwd=ROOT / "python"),
        run_timed(["../.venv/bin/python", "bench_onnx.py"], cwd=ROOT / "python"),
    ]

    footprint = {
        "native_binary_mb": (ROOT / "native" / "mlp").stat().st_size / 1e6,
        "torch_site_packages_mb": dir_size_mb(ROOT / ".venv/lib/python3.13/site-packages/torch"),
        "onnxruntime_site_packages_mb": dir_size_mb(
            ROOT / ".venv/lib/python3.13/site-packages/onnxruntime"
        ),
    }

    out = {"experiment": "exp1-tiny-mlp", "results": results, "dependency_footprint": footprint}
    print(json.dumps(out, indent=2))
    (ROOT / "research/experiments/exp1-tiny-mlp/results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
