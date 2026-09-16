"""exp15/16/17: the smallest real ONNX -> native C++ compiler.
exp15: Gemm->Relu->Gemm->Softmax, a fixed 4-op template, flat sequential IR.
exp16: generalized to arbitrary sequential chains of 5 ops (added LayerNorm/
GELU), still a flat IR where each op implicitly consumes "whatever the
previous op produced".
exp17 (this version): the IR became a real DAG. Each op now records its
ACTUAL named input tensor(s) and output tensor name, resolved from the ONNX
graph directly — not "the previous op's output". This is what residual
connections (y = LayerNorm(x + sublayer(x))) and, eventually, attention's
Q/K/V branching actually need: a tensor gets consumed more than once,
non-adjacently, which a flat chain has no way to represent.

Added op: Add (elementwise, exactly 2 non-initializer tensor inputs — a
residual connection, not a bias-add, which our Gemm nodes already fold in).

Codegen changed accordingly: instead of threading one "cur" buffer through
generated code, it declares one std::vector<float> per unique ONNX tensor
name that gets produced, and each op reads its named input variable(s) and
writes its named output variable — a real (if still restricted: no control
flow, no loops, fixed shapes) dataflow graph in C++, not just a chain.

Still narrow, stated explicitly: no attention yet (Q/K/V branching from one
input needs the same DAG machinery this experiment adds, but attention's
per-head reshape/transpose/batched-matmul is a different, larger step, not
attempted here). No opset<20 decomposed-GELU pattern-matching. No dynamic
shapes or control flow (If/Loop/Scan).
"""
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import onnx
from onnx import numpy_helper

NATIVE_DIR = pathlib.Path(__file__).parent.parent / "native"

SUPPORTED_OPS = {"Gemm", "Relu", "LayerNormalization", "Gelu", "Softmax", "Add"}


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
    graph_input_name = graph.input[0].name

    ir = []
    tensor_dims = {}  # ONNX tensor name -> channel dim, filled in as we discover it

    for node in graph.node:
        if node.op_type not in SUPPORTED_OPS:
            raise UnsupportedGraph(
                f"op '{node.op_type}' is not supported by this compiler "
                f"(supported: {sorted(SUPPORTED_OPS)})"
            )
        if len(node.output) != 1:
            raise UnsupportedGraph(f"op '{node.op_type}' has multiple outputs, not supported")

        non_init_inputs = [i for i in node.input if i not in initializer_names]
        out_name = node.output[0]
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}

        if node.op_type == "Gemm":
            if attrs.get("transB", 0) != 1 or attrs.get("alpha", 1.0) != 1.0 or attrs.get("beta", 1.0) != 1.0:
                raise UnsupportedGraph(f"Gemm with non-standard attrs (need transB=1, alpha=beta=1): {attrs}")
            if len(non_init_inputs) != 1:
                raise UnsupportedGraph(f"Gemm with {len(non_init_inputs)} tensor inputs, expected 1")
            in_tensor = non_init_inputs[0]
            _, w_name, b_name = node.input
            out_dim, in_dim = initializers[w_name].shape
            tensor_dims[in_tensor] = int(in_dim)  # learned retroactively from the weight shape
            tensor_dims[out_name] = int(out_dim)
            ir.append({"op": "linear", "output": out_name, "input": in_tensor,
                       "weight": w_name, "bias": b_name, "in_dim": int(in_dim), "out_dim": int(out_dim)})

        elif node.op_type in ("Relu", "Gelu", "Softmax"):
            if len(non_init_inputs) != 1:
                raise UnsupportedGraph(f"{node.op_type} with {len(non_init_inputs)} tensor inputs, expected 1")
            in_tensor = non_init_inputs[0]
            if node.op_type == "Gelu":
                approx = attrs.get("approximate", b"none")
                if approx not in (b"none", "none"):
                    raise UnsupportedGraph(f"Gelu approximate={approx!r} not supported — only exact/erf-based GELU")
            if node.op_type == "Softmax" and attrs.get("axis", -1) not in (-1, 1):
                raise UnsupportedGraph(f"Softmax over unsupported axis: {attrs.get('axis')}")
            op_name = {"Relu": "relu", "Gelu": "gelu", "Softmax": "softmax"}[node.op_type]
            tensor_dims[out_name] = tensor_dims[in_tensor]
            ir.append({"op": op_name, "output": out_name, "input": in_tensor, "dim": tensor_dims[in_tensor]})

        elif node.op_type == "LayerNormalization":
            if attrs.get("axis", -1) != -1:
                raise UnsupportedGraph(f"LayerNormalization over unsupported axis: {attrs.get('axis')}")
            if len(non_init_inputs) != 1:
                raise UnsupportedGraph(f"LayerNormalization with {len(non_init_inputs)} tensor inputs, expected 1")
            in_tensor = non_init_inputs[0]
            _, w_name, b_name = node.input
            dim = int(initializers[w_name].shape[0])
            eps = float(attrs.get("epsilon", 1e-5))
            tensor_dims[out_name] = dim
            ir.append({"op": "layernorm", "output": out_name, "input": in_tensor,
                       "weight": w_name, "bias": b_name, "dim": dim, "eps": eps})

        elif node.op_type == "Add":
            if len(non_init_inputs) != 2:
                raise UnsupportedGraph(
                    f"Add with {len(non_init_inputs)} non-initializer inputs, expected exactly 2 "
                    f"(a residual/skip connection) — bias-add is not this compiler's concern, "
                    f"Gemm nodes already fold their bias in"
                )
            a, b = non_init_inputs
            if a not in tensor_dims or b not in tensor_dims:
                raise UnsupportedGraph(f"Add operand dim not yet known (graph not in topological order?): {a}, {b}")
            if tensor_dims[a] != tensor_dims[b]:
                raise UnsupportedGraph(f"Add between mismatched dims: {tensor_dims[a]} vs {tensor_dims[b]}")
            tensor_dims[out_name] = tensor_dims[a]
            ir.append({"op": "add", "output": out_name, "inputs": [a, b], "dim": tensor_dims[a]})

    if not ir or ir[0]["op"] != "linear":
        raise UnsupportedGraph("this compiler requires the graph to start with a Linear (Gemm) layer")

    return ir, initializers, graph_input_name


def generate_cpp(ir, graph_input_name):
    """One block of C++ per IR op, reading/writing named variables that
    correspond directly to ONNX tensor names — a real dataflow graph, not a
    single threaded-through buffer. Declares one std::vector<float> per
    unique tensor produced; the graph's own input tensor maps to the
    function parameter X directly, no separate copy."""

    def var(tensor_name):
        return "X" if tensor_name == graph_input_name else cpp_id(tensor_name)

    input_dim = ir[0]["in_dim"]
    output_dim = next(op["out_dim"] for op in reversed(ir) if op["op"] == "linear")
    final_output_var = var(ir[-1]["output"])

    member_decls, load_lines, forward_lines = [], [], []

    for op in ir:
        out_var = var(op["output"])
        if op["op"] == "linear":
            w, b = cpp_id(op["weight"]), cpp_id(op["bias"])
            member_decls.append(f"    std::vector<float> {w}, {b};")
            load_lines.append(f'    m.{w} = load_f32(dir + "/{w}.bin", static_cast<size_t>({op["out_dim"]}) * {op["in_dim"]});')
            load_lines.append(f'    m.{b} = load_f32(dir + "/{b}.bin", {op["out_dim"]});')
            in_var = var(op["input"])
            forward_lines.append(f"        std::vector<float> {out_var}(static_cast<size_t>(n) * {op['out_dim']});")
            forward_lines.append(f"        linear({in_var}.data(), n, {op['in_dim']}, {w}.data(), {b}.data(), {op['out_dim']}, {out_var}.data());")
        elif op["op"] in ("relu", "gelu", "softmax"):
            in_var = var(op["input"])
            forward_lines.append(f"        std::vector<float> {out_var} = {in_var};")
            fn = {"relu": "relu_inplace", "gelu": "gelu_inplace", "softmax": None}[op["op"]]
            if op["op"] == "softmax":
                forward_lines.append(f"        softmax_rows({out_var}.data(), n, {op['dim']});")
            else:
                forward_lines.append(f"        {fn}({out_var}.data(), {out_var}.size());")
        elif op["op"] == "layernorm":
            w, b = cpp_id(op["weight"]), cpp_id(op["bias"])
            member_decls.append(f"    std::vector<float> {w}, {b};")
            load_lines.append(f'    m.{w} = load_f32(dir + "/{w}.bin", {op["dim"]});')
            load_lines.append(f'    m.{b} = load_f32(dir + "/{b}.bin", {op["dim"]});')
            in_var = var(op["input"])
            forward_lines.append(f"        std::vector<float> {out_var} = {in_var};")
            forward_lines.append(f"        layernorm_rows({out_var}.data(), n, {op['dim']}, {w}.data(), {b}.data(), {op['eps']}f);")
        elif op["op"] == "add":
            a_var, b_var = var(op["inputs"][0]), var(op["inputs"][1])
            forward_lines.append(f"        std::vector<float> {out_var} = {a_var};")
            forward_lines.append(f"        add_inplace({out_var}.data(), {b_var}.data(), {out_var}.size());")

    body = "\n".join(forward_lines)

    return f"""\
// AUTO-GENERATED by python/sajal_compile.py — do not hand-edit.
// Generated from an ONNX graph (DAG): {', '.join(f"{op['op']}->{cpp_id(op['output'])}" for op in ir)}.
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "common.hpp"

struct CompiledModel {{
    static constexpr int INPUT_DIM = {input_dim}, OUTPUT_DIM = {output_dim};
{chr(10).join(member_decls)}

    std::vector<float> forward(const std::vector<float>& X, int n) const {{
{body}
        return {final_output_var};
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

    ir, initializers, graph_input_name = parse_onnx(onnx_path)

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

    (out_dir / "model.cpp").write_text(generate_cpp(ir, graph_input_name))

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
