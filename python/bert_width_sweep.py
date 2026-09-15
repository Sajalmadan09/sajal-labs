"""exp13: exp12 bracketed the crossover between hidden=128 (1.40x native
edge) and hidden=256 (0.99x, no edge) but only measured those two
endpoints. This adds finer resolution — including a genuine midpoint
(192) and points beyond both ends (64, 320) — to see whether the
transition is a sharp threshold or a gradual slope, and to pin down where
native_vs_onnx actually crosses 1.0x. Fixed depth=1 throughout (isolates
width cleanly; exp12 already showed depth's effect is secondary and just
compounds whatever width already determined). head_dim=64 fixed as before
(multiples of 64 keep num_heads an integer and match how real BERT-family
models are actually configured — head_dim=64 is a near-universal
convention from bert-tiny up through much larger models).

Reuses exp12's export_config()/bench_one() machinery directly — this is
the same ablation, just more points on the width axis.
"""
import json
import pathlib

from bert_width_depth_sweep import bench_one

ROOT = pathlib.Path(__file__).parent.parent
WIDTHS = [64, 128, 192, 256, 320]  # multiples of 64 only, so num_heads=hidden/64 stays an exact integer and head_dim stays exactly 64 at every point — no new confound vs. exp12


def main():
    rows = [bench_one(h, 1) for h in WIDTHS]
    for r in rows:
        print(f"hidden={r['hidden']:>4}  native={r['native_p50_ms']:.5f}ms  onnx={r['onnx_p50_ms']:.5f}ms  "
              f"native_vs_onnx={r['native_vs_onnx']:.3f}x")

    # Linear-interpolate between measured points to estimate the crossover.
    crossover = None
    for a, b in zip(rows, rows[1:]):
        if (a["native_vs_onnx"] - 1.0) * (b["native_vs_onnx"] - 1.0) <= 0 and a["native_vs_onnx"] != b["native_vs_onnx"]:
            frac = (1.0 - a["native_vs_onnx"]) / (b["native_vs_onnx"] - a["native_vs_onnx"])
            crossover = a["hidden"] + frac * (b["hidden"] - a["hidden"])
            break

    print(f"\nestimated crossover (native_vs_onnx = 1.0x): hidden ~= {crossover:.0f}" if crossover
          else "\nno crossover found in this range")

    out = {"experiment": "exp13-width-sweep", "seq_len": 12, "num_layers": 1, "rows": rows,
           "estimated_crossover_hidden": crossover}
    out_dir = ROOT / "research/experiments/exp13-width-threshold"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
