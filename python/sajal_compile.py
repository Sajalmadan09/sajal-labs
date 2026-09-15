"""exp15/16: the smallest real ONNX -> native C++ compiler. exp15 supported
exactly Gemm->Relu->Gemm->Softmax (a fixed 4-op template); exp16 generalizes
this to arbitrary SEQUENTIAL chains of {Gemm, Relu, LayerNormalization,
Gelu, Softmax} — LayerNorm and GELU added, still no attention.

Deliberately still narrow, stated explicitly: "sequential chain" means each
op consumes exactly the previous op's output (plus weight initializers) and
produces exactly one new tensor. A residual/skip connection — the pattern
every real transformer block uses (x = LayerNorm(x + sublayer(x))) — has a
node with TWO non-initializer inputs (the sublayer output AND the original
x), which this compiler detects and explicitly refuses rather than silently
mishandling. Supporting that needs the IR to be a real DAG, not a flat list
— deferred to when attention is added (attention itself also branches: Q/K/V
all read the same input), not attempted here.

Codegen emits CALLS to common.hpp's existing, already-validated ops
(linear/relu_inplace/gelu_inplace/layernorm_rows/softmax_rows) — never a new
numerical kernel. Output artifact convention: weights extracted to
<sanitized-initializer-name>.bin files, dims baked into the generated code
as compile-time constants (no runtime shape parsing needed for the model
itself), plus a small test_config.txt (just "in_dim n_test") so the driver
knows how to size test_inputs.bin. This is intentionally NOT exp15's old
4-field shapes.txt convention — that convention coincidentally matched
mlp_model.hpp's fixed struct shape, which no longer holds once LayerNorm/
GELU are in the mix; reusing it here would let `sajal`'s existing "mlp kind"
detection silently misinterpret a richer graph as a plain MLP and compute
the wrong thing. Teaching `sajal` to recognize this newer, more general
convention is future work — not done here, stated as a real caveat.
"""
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import onnx
from onnx import numpy_helper

NATIVE_DIR = pathlib.Path(__file__).parent.parent / "native"

SUPPORTED_OPS = {"Gemm", "Relu", "LayerNormalization", "Gelu", "Softmax"}


class UnsupportedGraph(Exception):
    """Raised when the input ONNX graph is outside this compiler's
    deliberately narrow supported class — a clean, explicit failure is the
    correct behavior here, not a best-effort guess."""


def cpp_id(name):
    return "".join(c if c.isalnum() else "_" for c in name)


def parse_onnx(onnx_path):
    model = onnx.load(str(onnx_path))
    graph = model.graph
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    initializer_names = set(initializers.keys())

    ir = []
    current_tensor = graph.input[0].name

    for node in graph.node:
        if node.op_type not in SUPPORTED_OPS:
            raise UnsupportedGraph(
                f"op '{node.op_type}' is not supported by this compiler "
                f"(supported: {sorted(SUPPORTED_OPS)})"
            )

        non_init_inputs = [i for i in node.input if i not in initializer_names]
        if non_init_inputs != [current_tensor]:
            raise UnsupportedGraph(
                f"op '{node.op_type}' (output {node.output[0]!r}) has non-sequential inputs "
                f"{non_init_inputs} — expected exactly the previous op's output {current_tensor!r}. "
                f"This compiler only supports simple sequential chains, not branching/residual graphs "
                f"(e.g. LayerNorm(x + sublayer(x)) needs two inputs to the Add — not supported yet)."
            )
        if len(node.output) != 1:
            raise UnsupportedGraph(f"op '{node.op_type}' has multiple outputs, not supported")

        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}

        if node.op_type == "Gemm":
            if attrs.get("transB", 0) != 1 or attrs.get("alpha", 1.0) != 1.0 or attrs.get("beta", 1.0) != 1.0:
                raise UnsupportedGraph(f"Gemm with non-standard attrs (need transB=1, alpha=beta=1): {attrs}")
            _, w_name, b_name = node.input
            out_dim, in_dim = initializers[w_name].shape
            ir.append({"op": "linear", "weight": w_name, "bias": b_name,
                       "in_dim": int(in_dim), "out_dim": int(out_dim)})
        elif node.op_type == "Relu":
            ir.append({"op": "relu"})
        elif node.op_type == "LayerNormalization":
            if attrs.get("axis", -1) != -1:
                raise UnsupportedGraph(f"LayerNormalization over unsupported axis: {attrs.get('axis')}")
            _, w_name, b_name = node.input
            dim = int(initializers[w_name].shape[0])
            eps = float(attrs.get("epsilon", 1e-5))
            ir.append({"op": "layernorm", "weight": w_name, "bias": b_name, "dim": dim, "eps": eps})
        elif node.op_type == "Gelu":
            approx = attrs.get("approximate", b"none")
            if approx not in (b"none", "none"):
                raise UnsupportedGraph(
                    f"Gelu approximate={approx!r} not supported — only exact/erf-based GELU "
                    f"(matching common.hpp's gelu_inplace) is implemented"
                )
            ir.append({"op": "gelu"})
        elif node.op_type == "Softmax":
            axis = attrs.get("axis", -1)
            if axis not in (-1, 1):
                raise UnsupportedGraph(f"Softmax over unsupported axis: {axis}")
            ir.append({"op": "softmax"})

        current_tensor = node.output[0]

    if not ir or ir[0]["op"] != "linear":
        raise UnsupportedGraph("this compiler requires the graph to start with a Linear (Gemm) layer")

    return ir, initializers


def generate_cpp(ir):
    """One block of C++ per IR op, calling common.hpp's shared ops —
    genuinely generated from the IR list (length and contents vary per
    model), not filled into a fixed template."""
    input_dim = ir[0]["in_dim"]
    output_dim = next(op["out_dim"] for op in reversed(ir) if op["op"] == "linear")

    member_decls, load_lines, forward_lines = [], [], []
    forward_lines.append("        std::vector<float> cur = X;")
    forward_lines.append("        int cur_dim = INPUT_DIM;")

    for op in ir:
        if op["op"] == "linear":
            w, b = cpp_id(op["weight"]), cpp_id(op["bias"])
            member_decls.append(f"    std::vector<float> {w}, {b};")
            load_lines.append(f'    m.{w} = load_f32(dir + "/{w}.bin", static_cast<size_t>({op["out_dim"]}) * {op["in_dim"]});')
            load_lines.append(f'    m.{b} = load_f32(dir + "/{b}.bin", {op["out_dim"]});')
            forward_lines.append("        {")
            forward_lines.append(f"            std::vector<float> next(static_cast<size_t>(n) * {op['out_dim']});")
            forward_lines.append(f"            linear(cur.data(), n, cur_dim, {w}.data(), {b}.data(), {op['out_dim']}, next.data());")
            forward_lines.append(f"            cur = std::move(next); cur_dim = {op['out_dim']};")
            forward_lines.append("        }")
        elif op["op"] == "relu":
            forward_lines.append("        relu_inplace(cur.data(), cur.size());")
        elif op["op"] == "gelu":
            forward_lines.append("        gelu_inplace(cur.data(), cur.size());")
        elif op["op"] == "layernorm":
            w, b = cpp_id(op["weight"]), cpp_id(op["bias"])
            member_decls.append(f"    std::vector<float> {w}, {b};")
            load_lines.append(f'    m.{w} = load_f32(dir + "/{w}.bin", {op["dim"]});')
            load_lines.append(f'    m.{b} = load_f32(dir + "/{b}.bin", {op["dim"]});')
            forward_lines.append(f"        layernorm_rows(cur.data(), n, cur_dim, {w}.data(), {b}.data(), {op['eps']}f);")
        elif op["op"] == "softmax":
            forward_lines.append("        softmax_rows(cur.data(), n, cur_dim);")

    forward_lines.append("        return cur;")

    return f"""\
// AUTO-GENERATED by python/sajal_compile.py — do not hand-edit.
// Generated from an ONNX graph: {' -> '.join(op['op'] for op in ir)}.
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "common.hpp"

struct CompiledModel {{
    static constexpr int INPUT_DIM = {input_dim}, OUTPUT_DIM = {output_dim};
{chr(10).join(member_decls)}

    std::vector<float> forward(const std::vector<float>& X, int n) const {{
{chr(10).join(forward_lines)}
    }}
}};

CompiledModel load_model(const std::string& dir) {{
    CompiledModel m;
{chr(10).join(load_lines)}
    return m;
}}

void run_mode(const std::string& dir) {{
    CompiledModel m = load_model(dir);
    std::ifstream cfg(dir + "/test_config.txt");
    int in_dim, n_test;
    cfg >> in_dim >> n_test;
    auto X = load_f32(dir + "/test_inputs.bin", static_cast<size_t>(n_test) * in_dim);
    auto Y = m.forward(X, n_test);
    std::ofstream out(dir + "/native_outputs.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(Y.data()), Y.size() * sizeof(float));
    std::cerr << "wrote native_outputs.bin (" << n_test << "x" << CompiledModel::OUTPUT_DIM << ")\\n";
}}

void once_mode(const std::string& dir) {{
    CompiledModel m = load_model(dir);
    std::vector<float> x(CompiledModel::INPUT_DIM, 0.1f);
    auto y = m.forward(x, 1);
    asm volatile("" : : "g"(y.data()) : "memory");
}}

int main(int argc, char** argv) {{
    if (argc != 3) {{ std::cerr << "usage: " << argv[0] << " <dir> <run|once>\\n"; return 1; }}
    std::string dir = argv[1], mode = argv[2];
    if (mode == "run") run_mode(dir);
    else if (mode == "once") once_mode(dir);
    else {{ std::cerr << "unknown mode\\n"; return 1; }}
    return 0;
}}
"""


def compile_model(onnx_path, source_artifacts_dir, out_dir):
    """onnx_path: the .onnx graph to compile.
    source_artifacts_dir: where test_inputs.bin/ref_outputs.bin already
      live — copied through unchanged so equivalence is checked against the
      same reference the original experiment used, not freshly regenerated.
    out_dir: where the compiled artifact (weights, generated .cpp, compiled
      binary) is written.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_artifacts_dir = pathlib.Path(source_artifacts_dir)

    ir, initializers = parse_onnx(onnx_path)

    for op in ir:
        if op["op"] in ("linear", "layernorm"):
            initializers[op["weight"]].astype(np.float32).tofile(out_dir / f"{cpp_id(op['weight'])}.bin")
            initializers[op["bias"]].astype(np.float32).tofile(out_dir / f"{cpp_id(op['bias'])}.bin")

    if (source_artifacts_dir / "test_config.txt").exists():
        shutil.copy(source_artifacts_dir / "test_config.txt", out_dir / "test_config.txt")
    else:
        # exp1/exp9's older shapes.txt convention: "in_dim hidden_dim out_dim n_test"
        in_dim, _, _, n_test = (source_artifacts_dir / "shapes.txt").read_text().split()
        (out_dir / "test_config.txt").write_text(f"{in_dim} {n_test}\n")

    shutil.copy(source_artifacts_dir / "test_inputs.bin", out_dir / "test_inputs.bin")
    shutil.copy(source_artifacts_dir / "ref_outputs.bin", out_dir / "ref_outputs.bin")

    (out_dir / "model.cpp").write_text(generate_cpp(ir))

    binary_path = out_dir / "model"
    result = subprocess.run(
        ["clang++", "-O3", "-std=c++17", "-Wall", "-Wextra", "-DACCELERATE_NEW_LAPACK",
         f"-I{NATIVE_DIR}", "-framework", "Accelerate", "-o", str(binary_path), str(out_dir / "model.cpp")],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"generated code failed to compile:\n{result.stdout}\n{result.stderr}")

    return out_dir, binary_path


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} <model.onnx> <source_artifacts_dir> <out_dir>", file=sys.stderr)
        sys.exit(1)
    onnx_path, source_dir, out_dir_arg = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        out_dir, binary = compile_model(onnx_path, source_dir, out_dir_arg)
    except UnsupportedGraph as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"compiled {onnx_path} -> {binary}")
