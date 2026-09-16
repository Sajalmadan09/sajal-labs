"""exp15-22: the smallest real ONNX -> native C++ compiler.
exp15: Gemm->Relu->Gemm->Softmax, a fixed 4-op template, flat sequential IR.
exp16: generalized to 5 ops (added LayerNorm/GELU), still flat/sequential.
exp17: the IR became a real DAG (named tensors, not "previous op's output"),
added Add for residual connections.
exp18: single-head self-attention. NOT general MatMul/Transpose support —
this compiler recognizes exactly one 5-node pattern (Transpose(K,[1,0]) ->
MatMul(Q,K^T) -> Mul(scale) -> Softmax -> MatMul(V)) as a single fused
"self_attention" IR op, and rejects anything that doesn't match this exact
shape.
exp20 (this version): multi-head self-attention. PyTorch's exporter emits
the 3-way Reshape/Transpose head-split interleaved with the Q/K/V Gemm
nodes (not contiguous — a Gemm for K or V often sits between Q's Reshape
and the rest of the pattern), so this can't reuse exp18's contiguous
next-5-nodes lookahead. try_match_multihead_attention instead searches the
whole node list by tensor-name flow (who consumes whose output) to find the
12-node pattern regardless of position, then defers emitting its IR entry
until the main walk reaches the LAST consumed node index, so the generated
C++ still declares Q/K/V's linear outputs before the fused op uses them.

Codegen for both fused attention ops reuses the exact BLAS trick already
validated in native/transformer_model.hpp and native/bert_model.hpp: Q@K^T
via CblasTrans on K, no physical transpose; multi-head loops that per two
cblas_sgemm calls over head-sized column slices (pointer offset + full-row
stride, no physical splitting either). Not new numerical code — the same
validated pattern, now reachable from a compiled ONNX graph.

exp22: benchmarked the compiler against exp2's own hand-written baseline
(same architecture, weights, methodology) rather than only checking
correctness. Found a real gap by testing against exp2's actual model
instead of only synthetic test models: its attention scaling uses a `Div`
node (`x / sqrt(d)`), where every synthetic test model up to this point
happened to use `Mul` (`x * scale`) — mathematically identical, different
ONNX op. try_match_scale() now accepts either. Also added a `bench` mode
to generated code (matching transformer.cpp's methodology exactly) and
fixed `once_mode` to use the full sequence length for attention-containing
models rather than a hardcoded single row.

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


def try_match_scale(node, scores_tensor, constants):
    """Matches either `Mul(scores, c)` (scale by c) or `Div(scores, c)`
    (scale by 1/c) — mathematically the same "scale the attention scores"
    step, but two different ONNX ops depending on whether the exporting
    code wrote `x * (1/sqrt(d))` or `x / sqrt(d)`. exp2's own
    TinyTransformerBlock uses the latter; every synthetic compiler test
    model so far used the former — this one match covers both rather than
    requiring test models to happen to match the exact op a real model
    uses. Returns (scale_value, output_tensor) or None."""
    if node.op_type not in ("Mul", "Div") or scores_tensor not in node.input:
        return None
    candidates = [x for x in node.input if x != scores_tensor]
    if len(candidates) != 1 or candidates[0] not in constants:
        return None
    c = float(np.asarray(constants[candidates[0]]).reshape(-1)[0])
    scale_value = c if node.op_type == "Mul" else 1.0 / c
    return scale_value, node.output[0]


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

    scale_match = try_match_scale(n2, scores_tensor, constants)
    if scale_match is None:
        return None
    scale_value, scaled_tensor = scale_match

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


def try_match_multihead_attention(nodes, i, constants, tensor_dims, initializers):
    """Looks for the 12-node multi-head split/attend/merge pattern PyTorch's
    exporter emits for `q.view(seq,H,Dh).transpose(0,1)` etc: three
    (Reshape->Transpose) branches splitting Q/K/V into heads, MatMul->Mul->
    Softmax->MatMul attending per head (batched over the head dim), then a
    final Transpose->Reshape merging heads back to [seq,D]. Unlike
    try_match_self_attention, these nodes are NOT contiguous in nodes[i:] —
    PyTorch interleaves Q/K/V's Gemm nodes with the Reshape/Transpose nodes
    of branches parsed earlier — so this searches by tensor-name flow
    (who consumes whose output) rather than position, starting only when
    nodes[i] looks like the first half of one head-split branch.

    Returns (ir_entry, consumed_indices) or None. consumed_indices is the
    full set of node indices this match uses; the caller must defer
    appending ir_entry to the IR list until it reaches max(consumed_indices)
    — Q/K/V's own Gemm nodes (not part of consumed_indices) may sit at
    later positions than nodes[i], and the generated C++ must declare their
    outputs before the fused op reads them."""
    node0 = nodes[i]
    if node0.op_type != "Reshape" or node0.input[1] not in constants:
        return None
    shape0 = [int(x) for x in np.asarray(constants[node0.input[1]]).reshape(-1)]
    if len(shape0) != 3:
        return None
    num_heads, d_head = shape0[1], shape0[2]
    d = num_heads * d_head

    def resolve_dim(tensor_name):
        if tensor_name in tensor_dims:
            return tensor_dims[tensor_name]
        for n in nodes:
            if n.op_type == "Gemm" and list(n.output) == [tensor_name] and n.input[1] in initializers:
                return int(initializers[n.input[1]].shape[0])
        return None

    if resolve_dim(node0.input[0]) != d:
        return None

    def find_consumers(tensor_name, op_type):
        return [(idx, n) for idx, n in enumerate(nodes) if tensor_name in n.input and n.op_type == op_type]

    def perm_of(transpose_node):
        return next((list(a.ints) for a in transpose_node.attribute if a.name == "perm"), None)

    reshape_group = [
        idx for idx, n in enumerate(nodes)
        if n.op_type == "Reshape" and n.input[1] in constants
        and [int(x) for x in np.asarray(constants[n.input[1]]).reshape(-1)] == shape0
        and resolve_dim(n.input[0]) == d
    ]
    if len(reshape_group) != 3 or i not in reshape_group:
        return None

    branches = []
    for r_idx in reshape_group:
        r_node = nodes[r_idx]
        consumers = find_consumers(r_node.output[0], "Transpose")
        if len(consumers) != 1:
            return None
        t_idx, t_node = consumers[0]
        perm = perm_of(t_node)
        if perm is None or len(perm) != 3:
            return None
        branches.append({"reshape_idx": r_idx, "transpose_idx": t_idx, "perm": perm,
                          "out": t_node.output[0], "linear": r_node.input[0]})

    k_branches = [b for b in branches if b["perm"] == [1, 2, 0]]
    qv_branches = [b for b in branches if b["perm"] == [1, 0, 2]]
    if len(k_branches) != 1 or len(qv_branches) != 2:
        return None
    k_branch = k_branches[0]

    mm_pre = find_consumers(k_branch["out"], "MatMul")
    if len(mm_pre) != 1:
        return None
    mm_pre_idx, mm_pre_node = mm_pre[0]
    other = [x for x in mm_pre_node.input if x != k_branch["out"]]
    if len(other) != 1:
        return None
    q_branch = next((b for b in qv_branches if b["out"] == other[0]), None)
    if q_branch is None:
        return None
    v_branch = next(b for b in qv_branches if b is not q_branch)

    scale_matches = find_consumers(mm_pre_node.output[0], "Mul") + find_consumers(mm_pre_node.output[0], "Div")
    if len(scale_matches) != 1:
        return None
    mul_idx, mul_node = scale_matches[0]
    scale_match = try_match_scale(mul_node, mm_pre_node.output[0], constants)
    if scale_match is None:
        return None
    scale_value, scaled_tensor = scale_match

    softmax_matches = find_consumers(scaled_tensor, "Softmax")
    if len(softmax_matches) != 1:
        return None
    softmax_idx, softmax_node = softmax_matches[0]

    mm_post_matches = find_consumers(softmax_node.output[0], "MatMul")
    if len(mm_post_matches) != 1:
        return None
    mm_post_idx, mm_post_node = mm_post_matches[0]
    if [x for x in mm_post_node.input if x != softmax_node.output[0]] != [v_branch["out"]]:
        return None

    final_t_matches = find_consumers(mm_post_node.output[0], "Transpose")
    if len(final_t_matches) != 1:
        return None
    final_t_idx, final_t_node = final_t_matches[0]
    if perm_of(final_t_node) != [1, 0, 2]:
        return None

    final_r_matches = find_consumers(final_t_node.output[0], "Reshape")
    if len(final_r_matches) != 1:
        return None
    final_r_idx, final_r_node = final_r_matches[0]
    if final_r_node.input[1] not in constants:
        return None
    final_shape = [int(x) for x in np.asarray(constants[final_r_node.input[1]]).reshape(-1)]
    if len(final_shape) != 2 or final_shape[-1] != d:
        return None

    consumed = {b["reshape_idx"] for b in branches} | {b["transpose_idx"] for b in branches} | {
        mm_pre_idx, mul_idx, softmax_idx, mm_post_idx, final_t_idx, final_r_idx,
    }
    ir_entry = {"op": "multihead_self_attention", "q": q_branch["linear"], "k": k_branch["linear"],
                "v": v_branch["linear"], "num_heads": num_heads, "d_head": d_head,
                "scale": scale_value, "output": final_r_node.output[0], "dim": d}
    return ir_entry, consumed


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
    consumed_skip = set()
    pending_emit = {}
    i = 0
    while i < len(nodes):
        if i in pending_emit:
            ir_entry = pending_emit.pop(i)
            tensor_dims[ir_entry["output"]] = ir_entry["dim"]
            ir.append(ir_entry)
            i += 1
            continue
        if i in consumed_skip:
            i += 1
            continue
        node = nodes[i]

        match = try_match_self_attention(nodes, i, constants, tensor_dims)
        if match is not None:
            ir_entry, next_i = match
            tensor_dims[ir_entry["output"]] = ir_entry["dim"]
            ir.append(ir_entry)
            i = next_i
            continue

        mh_match = try_match_multihead_attention(nodes, i, constants, tensor_dims, initializers)
        if mh_match is not None:
            ir_entry, consumed = mh_match
            emit_at = max(consumed)
            consumed_skip |= consumed - {emit_at}
            pending_emit[emit_at] = ir_entry
            i += 1
            continue

        if node.op_type not in SUPPORTED_OPS:
            raise UnsupportedGraph(
                f"op '{node.op_type}' is not supported by this compiler outside the recognized "
                f"self-attention / multi-head-attention patterns "
                f"(supported standalone: {sorted(SUPPORTED_OPS)})"
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
        elif op["op"] == "multihead_self_attention":
            q_var, k_var, v_var = var(op["q"]), var(op["k"]), var(op["v"])
            d, dh, h_count, scale = op["dim"], op["d_head"], op["num_heads"], op["scale"]
            scores_var = out_var + "_scores"
            # Per-head loop with the same BLAS lda/ldb stride trick as
            # native/transformer_model.hpp: head h's Q/K/V is the column
            # slice [h*dh, (h+1)*dh) of the full [n,d] buffer, addressed by
            # pointer offset + full-row stride d — no physical splitting.
            forward_lines.append(f"        std::vector<float> {scores_var}(static_cast<size_t>(n) * n);")
            forward_lines.append(f"        std::vector<float> {out_var}(static_cast<size_t>(n) * {d});")
            forward_lines.append(f"        for (int h = 0; h < {h_count}; ++h) {{")
            forward_lines.append(
                f"            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, {dh}, {scale}f, "
                f"{q_var}.data() + h * {dh}, {d}, {k_var}.data() + h * {dh}, {d}, 0.0f, {scores_var}.data(), n);"
            )
            forward_lines.append(f"            softmax_rows({scores_var}.data(), n, n);")
            forward_lines.append(
                f"            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, {dh}, n, 1.0f, "
                f"{scores_var}.data(), n, {v_var}.data() + h * {dh}, {d}, 0.0f, {out_var}.data() + h * {dh}, {d});"
            )
            forward_lines.append("        }")

    body = "\n".join(forward_lines)

    # Attention-containing models mix information across all n rows of one
    # call (one sequence, not n independent examples — see
    # export_attention_test.py's docstring), so "one inference" for
    # bench/once means the full sequence length, read from test_config.txt
    # at runtime same as run_mode already does. Non-attention models keep
    # n=1 (one independent example), matching mlp.cpp/gender_predict.cpp's
    # own bench/once convention exactly, unchanged from exp15.
    has_attention = any(op["op"] in ("self_attention", "multihead_self_attention") for op in ir)
    if has_attention:
        read_n = '    std::ifstream cfg(dir + "/test_config.txt");\n    int in_dim, n;\n    cfg >> in_dim >> n;'
    else:
        read_n = "    int n = 1;"

    return f"""\
// AUTO-GENERATED by python/sajal_compile.py — do not hand-edit.
// Generated from an ONNX graph (DAG): {', '.join(f"{op['op']}->{cpp_id(op['output'])}" for op in ir)}.
#include <chrono>
#include <cstdio>
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

// Same methodology as native/transformer.cpp's bench_mode: cold_start_ms
// times load_model() alone, then 50 warmup + 500 measured warm-loop calls,
// same percentile() helper (common.hpp), same JSON print shape.
void bench_mode(const std::string& dir) {{
    auto t_start = Clock::now();
    CompiledModel m = load_model(dir);
    auto cold_start_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();

{read_n}
    std::vector<float> x(static_cast<size_t>(n) * CompiledModel::INPUT_DIM, 0.1f);

    const int WARMUP = 50, ITERS = 500;
    for (int i = 0; i < WARMUP; ++i) m.forward(x, n);

    std::vector<double> latencies_ms;
    latencies_ms.reserve(ITERS);
    for (int i = 0; i < ITERS; ++i) {{
        auto t0 = Clock::now();
        auto y = m.forward(x, n);
        latencies_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        asm volatile("" : : "g"(y.data()) : "memory");
    }}

    double mean = 0;
    for (double v : latencies_ms) mean += v;
    mean /= latencies_ms.size();

    std::printf(
        "{{\\"impl\\": \\"compiled_cpp\\", \\"cold_start_ms\\": %.4f, \\"warmup_iters\\": %d, \\"iters\\": %d, "
        "\\"mean_ms\\": %.5f, \\"p50_ms\\": %.5f, \\"p90_ms\\": %.5f, \\"p95_ms\\": %.5f, \\"p99_ms\\": %.5f}}\\n",
        cold_start_ms, WARMUP, ITERS, mean, percentile(latencies_ms, 50), percentile(latencies_ms, 90),
        percentile(latencies_ms, 95), percentile(latencies_ms, 99));
}}

void once_mode(const std::string& dir) {{
    CompiledModel m = load_model(dir);
{read_n}
    std::vector<float> x(static_cast<size_t>(n) * CompiledModel::INPUT_DIM, 0.1f);
    auto y = m.forward(x, n);
    asm volatile("" : : "g"(y.data()) : "memory");
}}

int main(int argc, char** argv) {{
    if (argc != 3) {{ std::cerr << "usage: " << argv[0] << " <dir> <run|bench|once>\\n"; return 1; }}
    std::string dir = argv[1], mode = argv[2];
    if (mode == "run") run_mode(dir);
    else if (mode == "bench") bench_mode(dir);
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
