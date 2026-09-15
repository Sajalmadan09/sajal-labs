"""exp14's benchmark: does paying real WordPiece tokenization cost (not
precomputed token IDs, unlike exp11) change the cold-invocation picture
for a real pretrained transformer? Same question exp10 answered for the
gender classifier's bigram features (answer there: no, negligible) — this
asks it again for a much heavier real tokenizer (30K-entry vocab, greedy
subword matching) on a much larger model.
"""
import json
import pathlib

from bench_common import percentiles, time_invocation

ROOT = pathlib.Path(__file__).parent.parent
TRIALS = 30


def main():
    py = str(ROOT / ".venv/bin/python")
    configs = [
        ("native_e2e", ["../native/bert", "../artifacts/bert_tiny", "once_e2e"]),
        ("pytorch_e2e", [py, "predict_bert_pytorch.py", "--once"]),
        ("onnx_e2e_transformers", [py, "predict_bert_onnx.py", "--once"]),
        ("onnx_e2e_lean", [py, "predict_bert_onnx_lean.py", "--once"]),
    ]

    wall_times = {name: [] for name, _ in configs}
    for _ in range(TRIALS):
        for name, cmd in configs:
            wall_times[name].append(time_invocation(cmd, cwd=ROOT / "python"))

    results = []
    for name, _ in configs:
        result = {"impl": name, "trials": TRIALS}
        result.update(percentiles(wall_times[name]))
        print(f"{name:>14}  p50={result['p50_ms']:.2f}ms  p95={result['p95_ms']:.2f}ms")
        results.append(result)

    out = {"experiment": "exp14-bert-e2e-cold-invocation", "trials": TRIALS, "results": results}
    out_dir = ROOT / "research/experiments/exp14-wordpiece-native"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cold_invocation_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_dir / 'cold_invocation_results.json'}")


if __name__ == "__main__":
    main()
