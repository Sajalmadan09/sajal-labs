"""exp15-18: the smallest real ONNX -> native C++ compiler.
exp15: Gemm->Relu->Gemm->Softmax, a fixed 4-op template, flat sequential IR.
exp16: generalized to 5 ops (added LayerNorm/GELU), still flat/sequential.
exp17: the IR became a real DAG (named tensors, not "previous op's output"),
added Add for residual connections.
exp18 (this version): single-head self-attention. NOT general MatMul/
Transpose support — this compiler recognizes exactly one 5-node pattern
(Transpose(K,[1,0]) -> MatMul(Q,K^T) -> Mul(scale) -> Softmax -> MatMul(V))
as a single fused "self_attention" IR op, and rejects anything that doesn't
match this exact shape. That's a deliberate, narrower claim than "supports
attention" — multi-head (reshape/transpose per head) is a separate, larger
step, not attempted here (see exp18's results.md).

Codegen for the fused op reuses the exact BLAS trick already validated in
native/transformer_model.hpp and native/bert_model.hpp: Q@K^T is one
cblas_sgemm call with CblasTrans on K (no physical transpose, no separate
IR/codegen handling for the Transpose node — it's consumed entirely inside
the fusion), scaled attention @ V is a second cblas_sgemm call. This is not
new numerical code, it's the same validated pattern, now reachable from a
compiled ONNX graph instead of only from hand-written headers.

Everything else (op-to-function mapping, DAG-based tensor tracking, weight
extraction convention) is unchanged from exp17 — see its docstring.
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


def try_match_self_attention(nodes, i, constants, tensor_dims):
    """Looks for exactly: Transpose(K,[1,0]) -> MatMul(Q,K^T) -> Mul(scale)
    -> Softmax -> MatMul(attn,V), starting at nodes[i]. Returns
    (ir_entry, next_index) or None — a pattern match, not general MatMul/
    Transpose support; anything not shaped exactly like this falls through
    to the caller's normal per-node handling (which will reject Transpose/
    MatMul as unsupported op types on their own)."""
    if i + 4 >= len(nodes):
        return None
    n0, n1, n2, n3, n4 = nodes[i:i + 5]

    if n0.op_type != "Transpose":
        return None
    perm = next((list(onnx.helper.get_attribute_value(a)) for a in n0.attribute if a.name == "perm"), None)
    if perm != [1, 0]:
        return None
    k_tensor, kt_tensor = n0.input[0], n0.output[0]

    if n1.op_type != "MatMul" or kt_tensor not in n1.input:
        return None
    others = [x for x in n1.input if x != kt_tensor]
    if len(others) != 1:
        return None
    q_tensor, scores_tensor = others[0], n1.output[0]

    if n2.op_type != "Mul" or scores_tensor not in n2.input:
        return None
    scale_candidates = [x for x in n2.input if x != scores_tensor]
    if len(scale_candidates) != 1 or scale_candidates[0] not in constants:
        return None
    scale_value = float(np.asarray(constants[scale_candidates[0]]).reshape(-1)[0])
    scaled_tensor = n2.output[0]

    if n3.op_type != "Softmax" or list(n3.input) != [scaled_tensor]:
        return None
    attn_tensor = n3.output[0]

    if n4.op_type != "MatMul" or attn_tensor not in n4.input:
        return None
    v_candidates = [x for x in n4.input if x != attn_tensor]
    if len(v_candidates) != 1:
        return None
    v_tensor = v_candidates[0]

    if q_tensor not in tensor_dims or k_tensor not in tensor_dims or v_tensor not in tensor_dims:
        return None
    d = tensor_dims[q_tensor]
    if tensor_dims[k_tensor] != d or tensor_dims[v_tensor] != d:
        return None

    ir_entry = {"op": "self_attention", "q": q_tensor, "k": k_tensor, "v": v_tensor,
                "scale": scale_value, "output": n4.output[0], "dim": d}
    return ir_entry, i + 5


def parse_onnx(onnx_path):
    model = onnx.load(str(onnx_path))
    graph = model.graph
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    initializer_names = set(initializers.keys())
    graph_input_name = graph.input[0].name

    # Constant nodes (e.g. the attention scale factor) are pre-resolved to
    # values and skipped in the main walk — they carry no runtime input.
    constants = {}
    nodes = []
    for node in graph.node:
        if node.op_type == "Constant":
            val_attr = next(a for a in node.attribute if a.name == "value")
            constants[node.output[0]] = numpy_helper.to_array(onnx.helper.get_attribute_value(val_attr))
        else:
            nodes.append(node)

    ir = []
    tensor_dims = {}
    i = 0
    while i < len(nodes):
        node = nodes[i]

        match = try_match_self_attention(nodes, i, constants, tensor_dims)
        if match is not None:
            ir_entry, next_i = match
            tensor_dims[ir_entry["output"]] = ir_entry["dim"]
            ir.append(ir_entry)
            i = next_i
            continue

        if node.op_type not in SUPPORTED_OPS:
            raise UnsupportedGraph(
                f"op '{node.op_type}' is not supported by this compiler outside the recognized "
                f"self-attention pattern (supported standalone: {sorted(SUPPORTED_OPS)})"
            )
        if len(node.output) != 1:
            raise UnsupportedGraph(f"op '{node.op_type}' has multiple outputs, not supported")

        non_init_inputs = [x for x in node.input if x not in initializer_names]
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
            tensor_dims[in_tensor] = int(in_dim)
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

        i += 1

    if not ir or ir[0]["op"] != "linear":
        raise UnsupportedGraph("this compiler requires the graph to start with a Linear (Gemm) layer")

    return ir, initializers, graph_input_name


def generate_cpp(ir, graph_input_name):
    """One block of C++ per IR op, reading/writing named variables that
    correspond directly to ONNX tensor names."""

    def var(tensor_name):
        return "X" if tensor_name == graph_input_name else cpp_id(tensor_name)

    input_dim = ir[0]["in_dim"]
    # Last Linear's out_dim — correct for every model tested so far (all end in a
    # projection). A model ending directly in attention/LayerNorm with no trailing
    # Linear would need this generalized; not attempted since no test case needs it yet.
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
            if op["op"] == "softmax":
                forward_lines.append(f"        softmax_rows({out_var}.data(), n, {op['dim']});")
            else:
                fn = "relu_inplace" if op["op"] == "relu" else "gelu_inplace"
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
        elif op["op"] == "self_attention":
            q_var, k_var, v_var = var(op["q"]), var(op["k"]), var(op["v"])
            d, scale = op["dim"], op["scale"]
            scores_var = out_var + "_scores"
            # Same BLAS trick as native/transformer_model.hpp and native/bert_model.hpp:
            # CblasTrans on K computes Q@K^T with no physical transpose.
            forward_lines.append(f"        std::vector<float> {scores_var}(static_cast<size_t>(n) * n);")
            forward_lines.append(
                f"        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, {d}, {scale}f, "
                f"{q_var}.data(), {d}, {k_var}.data(), {d}, 0.0f, {scores_var}.data(), n);"
            )
            forward_lines.append(f"        softmax_rows({scores_var}.data(), n, n);")
            forward_lines.append(f"        std::vector<float> {out_var}(static_cast<size_t>(n) * {d});")
            forward_lines.append(
                f"        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, {d}, n, 1.0f, "
                f"{scores_var}.data(), n, {v_var}.data(), {d}, 0.0f, {out_var}.data(), {d});"
            )

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
