"""Experiment 7: the pivot. exp1-6 chased warm per-call latency and found
native C++ never beat ONNX Runtime CPU on any transformer-shaped workload
(exp2-6) — only exp1's 2-matmul MLP showed a real warm-latency win. What DID
hold up, every single time, at every size: cold-start and dependency
footprint (research/gap-analysis.md's revised recommendation after exp6).

This measures the metric that pivot actually claims to win on: true
end-to-end COLD INVOCATION latency — process spawn to answer to process
exit, timed from OUTSIDE the process (wall-clock wrapping each subprocess
call), for one single inference. This is what a serverless function, a CLI
tool, or an edge device invoked occasionally actually pays — not the warm
in-process loop exp1-6 measured. Each trial is a genuinely fresh OS process
(not a reused one), which is the honest way to measure this.
"""
import json
import pathlib
import time

from bench_common import percentiles, dir_size_mb

ROOT = pathlib.Path(__file__).parent.parent
TRIALS = 30


def time_invocation(cmd, cwd):
    import subprocess
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd} failed:\n{proc.stdout}\n{proc.stderr}")
    return wall_ms


def main():
    py = str(ROOT / ".venv/bin/python")
    configs = [
        ("native_mlp", ["../native/mlp", "../artifacts", "once"]),
        ("pytorch_mlp", [py, "bench_pytorch.py", "--once"]),
        ("onnx_mlp", [py, "bench_onnx.py", "--once"]),
        ("native_transformer", ["../native/transformer", "../artifacts/transformer", "once"]),
        ("pytorch_transformer", [py, "bench_pytorch_transformer.py", "--once"]),
        ("onnx_transformer", [py, "bench_onnx_transformer.py", "--once"]),
    ]

    # Interleaved round-robin, not one config's 30 trials back-to-back: two
    # configs sharing a library (e.g. onnx_mlp, onnx_transformer both import
    # onnxruntime) would otherwise let whichever runs first warm the OS page
    # cache for the other's shared libraries, biasing the comparison.
    wall_times = {name: [] for name, _ in configs}
    for trial in range(TRIALS):
        for name, cmd in configs:
            wall_times[name].append(time_invocation(cmd, cwd=ROOT / "python"))

    results = []
    for name, _ in configs:
        result = {"impl": name, "trials": TRIALS}
        result.update(percentiles(wall_times[name]))
        print(f"{name:>20}  p50={result['p50_ms']:.2f}ms  p95={result['p95_ms']:.2f}ms")
        results.append(result)

    footprint = {
        "native_mlp_binary_mb": (ROOT / "native" / "mlp").stat().st_size / 1e6,
        "native_transformer_binary_mb": (ROOT / "native" / "transformer").stat().st_size / 1e6,
        "torch_site_packages_mb": dir_size_mb(ROOT / ".venv/lib/python3.13/site-packages/torch"),
        "onnxruntime_site_packages_mb": dir_size_mb(
            ROOT / ".venv/lib/python3.13/site-packages/onnxruntime"
        ),
    }

    out = {"experiment": "exp7-cold-invocation", "trials": TRIALS, "results": results,
           "dependency_footprint": footprint}
    out_dir = ROOT / "research/experiments/exp7-cold-invocation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
