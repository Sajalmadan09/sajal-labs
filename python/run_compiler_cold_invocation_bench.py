"""exp22: does the COMPILED artifact (sajal_compile.py's auto-generated
code) cost anything vs. exp2's hand-written transformer.cpp, at exp2's
actual benchmarked scale (seq_len=32, d_model=256, n_heads=4, d_ff=1024)?
Same cold-invocation methodology as exp7/8 (bench_common.time_invocation:
wall-clock wrapping each subprocess call from outside, genuinely fresh OS
process per trial, interleaved round-robin so neither binary gets an
unfair page-cache-warming advantage from running first).
"""
import json
import pathlib

from bench_common import percentiles, time_invocation

ROOT = pathlib.Path(__file__).parent.parent
TRIALS = 30


def main():
    configs = [
        ("handwritten_transformer", ["../native/transformer", "../artifacts/exp2_scale_test", "once"]),
        ("compiled_transformer", ["../artifacts/compiled/exp2_scale_test/model", "../artifacts/compiled/exp2_scale_test", "once"]),
    ]

    wall_times = {name: [] for name, _ in configs}
    for trial in range(TRIALS):
        for name, cmd in configs:
            wall_times[name].append(time_invocation(cmd, cwd=ROOT / "python"))

    results = []
    for name, _ in configs:
        result = {"impl": name, "trials": TRIALS}
        result.update(percentiles(wall_times[name]))
        print(f"{name:>24}  p50={result['p50_ms']:.3f}ms  p95={result['p95_ms']:.3f}ms")
        results.append(result)

    footprint = {
        "handwritten_transformer_binary_mb": (ROOT / "native" / "transformer").stat().st_size / 1e6,
        "compiled_transformer_binary_mb": (ROOT / "artifacts" / "compiled" / "exp2_scale_test" / "model").stat().st_size / 1e6,
    }

    out = {"experiment": "exp22-compiler-vs-handwritten-baseline", "trials": TRIALS,
           "results": results, "footprint": footprint}
    out_path = ROOT / "research" / "experiments" / "exp22-compiler-vs-baseline" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    print(json.dumps(footprint, indent=2))


if __name__ == "__main__":
    main()
