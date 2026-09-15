"""exp12: exp11's warm-loop result didn't fit exp5's "operator count, not
language" theory cleanly — bert-tiny (hidden=128, 2 layers) retained a
modest native edge (1.25x vs ONNX) while exp2's synthetic transformer
(hidden=256, 1 layer) collapsed to ~1.0x, despite bert-tiny making MORE
total BLAS calls (2 layers' worth). But that comparison has confounds:
exp2 used TransformerBlock + the legacy ONNX exporter; exp11 used
bert_model.hpp + the dynamo exporter. Different toolchains could explain
the difference as easily as different architecture shape.

This sweep isolates architecture shape as the ONLY variable: every config
here goes through bert_model.hpp and bert_common.py's exporter (the exact
same toolchain exp11 used), varying only hidden_size (128 vs 256, matching
exp11 vs exp2's widths) and num_layers (1 vs 2, matching exp2 vs exp11's
depths), holding head_dim=64 fixed (matches both exp2 and exp11 exactly)
and seq_len=12 fixed (matches exp11's actual benchmark case).

Random (not pretrained) weights — this sweep is purely about latency, not
accuracy, so a real pretrained checkpoint isn't needed and doesn't exist
for arbitrary (hidden, layers) combinations anyway.
"""
import json
import pathlib

import torch
from transformers import BertConfig, BertModel

from bench_common import run_timed
from bert_common import export_bert_native_weights, export_bert_onnx, export_case

ROOT = pathlib.Path(__file__).parent.parent
SEQ_LEN = 12
VOCAB_SIZE = 1000  # small synthetic vocab — no real tokenizer needed, token ids are arbitrary indices
CONFIGS = [(128, 1), (128, 2), (256, 1), (256, 2)]  # (hidden_size, num_layers), head_dim=64 throughout


def export_config(hidden, num_layers, artifacts_dir):
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    config = BertConfig(hidden_size=hidden, num_hidden_layers=num_layers, num_attention_heads=hidden // 64,
                         intermediate_size=4 * hidden, vocab_size=VOCAB_SIZE, max_position_embeddings=64)
    model = BertModel(config).eval()  # random init, not from_pretrained

    export_bert_native_weights(model, artifacts_dir)

    torch.manual_seed(1)
    dummy_ids = torch.randint(0, VOCAB_SIZE, (3,))
    export_case(model, artifacts_dir, 0, dummy_ids)  # short dummy case (bert.cpp's convention: case0 unused by bench)
    bench_ids = torch.randint(0, VOCAB_SIZE, (SEQ_LEN,))
    bench_encoded = export_case(model, artifacts_dir, 1, bench_ids)  # bert.cpp's bench_mode always reads case1

    (artifacts_dir / "test_cases.txt").write_text(f"3 dummy\n{SEQ_LEN} bench case\n")
    export_bert_onnx(model, artifacts_dir, bench_encoded)
    return model, config


def bench_one(hidden, num_layers):
    tag = f"h{hidden}_l{num_layers}"
    artifacts_dir = ROOT / "artifacts" / "bert_sweep" / tag
    export_config(hidden, num_layers, artifacts_dir)

    native = run_timed(["../native/bert", str(artifacts_dir), "bench"], cwd=ROOT / "python")
    onnx = run_timed(["../.venv/bin/python", "bench_sweep_bert_onnx.py", str(artifacts_dir)], cwd=ROOT / "python")

    return {
        "hidden": hidden, "num_layers": num_layers, "head_dim": hidden // (hidden // 64),
        "native_p50_ms": native["p50_ms"], "onnx_p50_ms": onnx["p50_ms"],
        "native_vs_onnx": onnx["p50_ms"] / native["p50_ms"],
    }


def main():
    rows = [bench_one(h, l) for h, l in CONFIGS]
    for r in rows:
        print(f"hidden={r['hidden']:>4} layers={r['num_layers']}  native={r['native_p50_ms']:.5f}ms  "
              f"onnx={r['onnx_p50_ms']:.5f}ms  native_vs_onnx={r['native_vs_onnx']:.2f}x")

    out = {"experiment": "exp12-width-depth-sweep", "seq_len": SEQ_LEN, "rows": rows}
    out_dir = ROOT / "research/experiments/exp12-warm-latency-width-depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
