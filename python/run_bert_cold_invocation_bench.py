"""exp11's benchmark: does the cold-invocation advantage (exp7-10) hold for
a real, much larger (4.4M param) pretrained transformer with a real
tokenizer/library stack behind the Python side (transformers, not just
torch)? Same methodology as exp7-10: external wall-clock, interleaved
trials, `once` mode.
"""
import json
import pathlib

from bench_common import percentiles, time_invocation

ROOT = pathlib.Path(__file__).parent.parent
TRIALS = 30


def main():
    py = str(ROOT / ".venv/bin/python")
    configs = [
        ("native", ["../native/bert", "../artifacts/bert_tiny", "once"]),
        ("pytorch_transformers", [py, "bench_bert_pytorch.py", "--once"]),
        ("onnx", [py, "bench_bert_onnx.py", "--once"]),
    ]

    wall_times = {name: [] for name, _ in configs}
    for _ in range(TRIALS):
        for name, cmd in configs:
            wall_times[name].append(time_invocation(cmd, cwd=ROOT / "python"))

    results = []
    for name, _ in configs:
        result = {"impl": name, "trials": TRIALS}
        result.update(percentiles(wall_times[name]))
        print(f"{name:>22}  p50={result['p50_ms']:.2f}ms  p95={result['p95_ms']:.2f}ms")
        results.append(result)

    out = {"experiment": "exp11-bert-cold-invocation", "trials": TRIALS, "results": results}
    out_dir = ROOT / "research/experiments/exp11-real-pretrained-transformer"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cold_invocation_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_dir / 'cold_invocation_results.json'}")


if __name__ == "__main__":
    main()
