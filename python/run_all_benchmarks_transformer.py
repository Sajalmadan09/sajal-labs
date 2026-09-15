import json
import pathlib

from bench_common import run_timed, dir_size_mb

ROOT = pathlib.Path(__file__).parent.parent


def main():
    results = [
        run_timed(["../native/transformer", "../artifacts/transformer", "bench"], cwd=ROOT / "python"),
        run_timed(["../.venv/bin/python", "bench_pytorch_transformer.py"], cwd=ROOT / "python"),
        run_timed(["../.venv/bin/python", "bench_onnx_transformer.py"], cwd=ROOT / "python"),
    ]

    footprint = {
        "native_binary_mb": (ROOT / "native" / "transformer").stat().st_size / 1e6,
        "torch_site_packages_mb": dir_size_mb(ROOT / ".venv/lib/python3.13/site-packages/torch"),
        "onnxruntime_site_packages_mb": dir_size_mb(
            ROOT / ".venv/lib/python3.13/site-packages/onnxruntime"
        ),
    }

    out = {"experiment": "exp2-tiny-transformer", "results": results, "dependency_footprint": footprint}
    print(json.dumps(out, indent=2))
    (ROOT / "research/experiments/exp2-tiny-transformer/results.json").write_text(
        json.dumps(out, indent=2)
    )


if __name__ == "__main__":
    main()
