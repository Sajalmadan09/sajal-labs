"""exp15-28: the smallest real ONNX -> native C++ compiler.
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

exp23: stacking multiple encoder blocks in one graph exposed a real bug in
exp20's original multi-head matcher — it found its Q/K/V branches via a
global scan for reshapes matching a (num_heads, d_head) shape, which
silently merges branches from DIFFERENT attention instances once a graph
contains more than one (every prior experiment had exactly one, so this
never showed up). Rewrote try_match_multihead_attention to be purely
tensor-flow-traced from wherever it starts — see its docstring.

exp24: widened GELU support to the 5-node DECOMPOSED form ONNX exporters
emit at opset<20 (`Gelu` only became a fused op at opset 20) —
`0.5*x*(1+erf(x/sqrt(2)))` as `Div->Erf->Add->Mul->Mul`. Unlike multi-head
attention's Q/K/V split, this pattern has no parallel branches (it's a
strictly sequential chain), so a positional lookahead (like exp18's
try_match_self_attention) is safe here — exp23's tensor-flow-tracing
rewrite was needed specifically because attention's branches interleave
with other nodes; a linear chain never does.

exp25: tried the compiler against prajjwal1/bert-tiny's REAL export
(dynamo=False, HF's own BertLayer.forward()) — found 4 real gaps
(MatMul+Add instead of Gemm, 4D-batched attention, split pre-matmul
scaling, an unexplained Gather) and routed around them by rewiring the
real weights into this project's own known convention instead.

exp26: closed two of those four gaps directly, rather than continuing to
route around them — see try_match_matmul_add_linear and
try_match_multihead_attention's generalized shape/scale handling below.
Result: bert-tiny's real 2-layer encoder, traced by HF's own code with
ZERO rewiring, now compiles and matches HF's actual forward pass to
2.86e-06 — a stronger result than exp25's rewired version. Also the first
compiler target that isn't bit-identical to a hand-written reference (see
exp26/results.md) — traced to combining two independently-rounded scale
factors via Python float multiplication, not a bug: the exact IEEE-754
non-associativity caveat this project's methodology has named since exp1,
finally encountered in practice.

exp27: causal (autoregressive) attention masking — recognizes PyTorch's
standard `scores.masked_fill(causal_mask, -inf)` export
(`Trilu(ones,k=1)->Cast->Where`) sitting between the scale step and
Softmax in both try_match_self_attention and
try_match_multihead_attention. A modifier on the existing attention IR
ops (`causal: bool`), not a new op — codegen just runs
common.hpp's causal_mask_rows() before softmax_rows(), reusing that its
-inf entries already drop out of softmax's row sum correctly.

exp28: a real GPT-2 decoder block surfaced six differences from every
encoder model tried through exp27. Three closed directly:
try_match_identity_reshape (Conv1D's flatten/unflatten Reshape wrapping —
a no-op under this project's batch=1 convention), Gemm transB=0 support
(Conv1D's weight stored [in,out], reusing exp26's weight_transposed
mechanism), and gelu_tanh as a genuinely SEPARATE IR op/C++ primitive
from exact "gelu" (GPT-2's "gelu_new" activation is different math, not
a different encoding of the same math — conflating them would have been
a silent correctness bug). Two turned out to need nothing beyond a
one-line dimension-bookkeeping fix: pre-norm ordering (LayerNorm before
the sub-block) exposed that LayerNormalization's handler never recorded
its OWN input's dim (only its output's) — every prior model was
post-norm, where some earlier Linear had always already recorded it —
and the "graph must start with Linear" check was simply too strict for a
graph that starts with LayerNorm instead. One gap (Q/K/V from a single
combined Conv1D via ONNX Split) was named and routed around rather than
closed, same as exp25's handling of BERT's stray Gather.

Everything else (op-to-function mapping, DAG-based tensor tracking, weight
extraction convention) is unchanged from exp17 — see its docstring.
"""
import math
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


def _constant_value(tensor_name, constants):
    if tensor_name not in constants:
        return None
    return float(np.asarray(constants[tensor_name]).reshape(-1)[0])


def try_match_decomposed_gelu(nodes, i, constants, tensor_dims):
    """Matches the 5-node decomposed GELU ONNX exporters emit at opset<20
    (before `Gelu` existed as a fused op): `0.5*x*(1+erf(x/sqrt(2)))` as
    `Div(x,sqrt2) -> Erf -> Add(1) -> Mul(x,·) -> Mul(0.5,·)`. Strictly
    sequential (no parallel branches to interleave with other nodes, unlike
    multi-head attention's Q/K/V split), so contiguous positional lookahead
    is safe — same style as try_match_self_attention, not
    try_match_multihead_attention's tensor-flow tracing.
    Returns (ir_entry, next_index) or None."""
    if i + 4 >= len(nodes):
        return None
    n0, n1, n2, n3, n4 = nodes[i:i + 5]

    if n0.op_type != "Div":
        return None
    x_tensor = n0.input[0]
    c1_candidates = [t for t in n0.input if t != x_tensor]
    if len(c1_candidates) != 1:
        return None
    c1 = _constant_value(c1_candidates[0], constants)
    if c1 is None or not math.isclose(c1, math.sqrt(2.0), rel_tol=1e-4):
        return None

    if n1.op_type != "Erf" or list(n1.input) != [n0.output[0]]:
        return None

    if n2.op_type != "Add" or n1.output[0] not in n2.input:
        return None
    c2_candidates = [t for t in n2.input if t != n1.output[0]]
    if len(c2_candidates) != 1:
        return None
    c2 = _constant_value(c2_candidates[0], constants)
    if c2 is None or not math.isclose(c2, 1.0, rel_tol=1e-4):
        return None

    if n3.op_type != "Mul" or sorted(n3.input) != sorted([x_tensor, n2.output[0]]):
        return None

    if n4.op_type != "Mul" or n3.output[0] not in n4.input:
        return None
    c3_candidates = [t for t in n4.input if t != n3.output[0]]
    if len(c3_candidates) != 1:
        return None
    c3 = _constant_value(c3_candidates[0], constants)
    if c3 is None or not math.isclose(c3, 0.5, rel_tol=1e-4):
        return None

    if x_tensor not in tensor_dims:
        return None
    return {"op": "gelu", "output": n4.output[0], "input": x_tensor, "dim": tensor_dims[x_tensor]}, i + 5


def try_match_tanh_gelu(nodes, i, constants, tensor_dims):
    """Matches the 8-node "gelu_new" (tanh-approximation) GELU GPT-2's own
    code uses (transformers.activations.NewGELUActivation), a DIFFERENT
    decomposition from try_match_decomposed_gelu's exact/erf formula:
    `0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))` as
    `Mul(x,0.5) ; Pow(x,3) ; Mul(·,0.044715) ; Add(x,·) ; Mul(·,sqrt(2/pi))
    ; Tanh ; Add(·,1) ; Mul(half_x,·)`. Strictly sequential (like
    try_match_decomposed_gelu), so positional lookahead is safe here too.
    Returns (ir_entry, next_index) or None."""
    if i + 7 >= len(nodes):
        return None
    n0, n1, n2, n3, n4, n5, n6, n7 = nodes[i:i + 8]

    if n0.op_type != "Mul":
        return None
    x_candidates = [t for t in n0.input if t not in constants]
    if len(x_candidates) != 1:
        return None
    x_tensor = x_candidates[0]
    c0 = _constant_value([t for t in n0.input if t != x_tensor][0], constants)
    if c0 is None or not math.isclose(c0, 0.5, rel_tol=1e-4):
        return None
    half_x = n0.output[0]

    if n1.op_type != "Pow" or list(n1.input)[0] != x_tensor:
        return None
    c1 = _constant_value(n1.input[1], constants)
    if c1 is None or not math.isclose(c1, 3.0, rel_tol=1e-4):
        return None

    if n2.op_type != "Mul" or n1.output[0] not in n2.input:
        return None
    c2_candidates = [t for t in n2.input if t != n1.output[0]]
    if len(c2_candidates) != 1:
        return None
    c2 = _constant_value(c2_candidates[0], constants)
    if c2 is None or not math.isclose(c2, 0.044715, rel_tol=1e-3):
        return None

    if n3.op_type != "Add" or sorted(n3.input) != sorted([x_tensor, n2.output[0]]):
        return None

    if n4.op_type != "Mul" or n3.output[0] not in n4.input:
        return None
    c4_candidates = [t for t in n4.input if t != n3.output[0]]
    if len(c4_candidates) != 1:
        return None
    c4 = _constant_value(c4_candidates[0], constants)
    if c4 is None or not math.isclose(c4, math.sqrt(2.0 / math.pi), rel_tol=1e-4):
        return None

    if n5.op_type != "Tanh" or list(n5.input) != [n4.output[0]]:
        return None

    if n6.op_type != "Add" or n5.output[0] not in n6.input:
        return None
    c6_candidates = [t for t in n6.input if t != n5.output[0]]
    if len(c6_candidates) != 1:
        return None
    c6 = _constant_value(c6_candidates[0], constants)
    if c6 is None or not math.isclose(c6, 1.0, rel_tol=1e-4):
        return None

    if n7.op_type != "Mul" or sorted(n7.input) != sorted([half_x, n6.output[0]]):
        return None

    if x_tensor not in tensor_dims:
        return None
    return {"op": "gelu_tanh", "output": n7.output[0], "input": x_tensor, "dim": tensor_dims[x_tensor]}, i + 8


def try_match_identity_reshape(node, constants, tensor_dims):
    """Recognizes a Reshape that only adds or removes a leading batch=1
    dimension around an otherwise-unchanged [n, feature_dim] tensor — e.g.
    GPT-2's Conv1D (exp28) explicitly flattens to 2D before its matmul and
    back to 3D after (`x.view(-1, x.size(-1))` ... `x.view(size_out)`),
    unlike nn.Linear which never surfaces this as a visible op. Structurally
    a no-op under this project's batch=1 convention (every op already
    treats n as a runtime row count, and this project has never modeled a
    real batch dimension) — passed through as a copy rather than needing
    any new numeric capability. Returns an ir_entry or None."""
    if node.op_type != "Reshape" or node.input[1] not in constants:
        return None
    shape = [int(x) for x in np.asarray(constants[node.input[1]]).reshape(-1)]
    if not shape:
        return None
    d = shape[-1]
    if d <= 0:
        return None
    middle = shape[:-1]
    while middle and middle[0] == 1:
        middle = middle[1:]
    if len(middle) > 1:
        return None  # more than one real (non-batch) dim besides features
    in_tensor = node.input[0]
    if in_tensor not in tensor_dims or tensor_dims[in_tensor] != d:
        return None
    return {"op": "identity", "output": node.output[0], "input": in_tensor, "dim": d}


def try_match_matmul_add_linear(nodes, i, initializer_names, initializers):
    """Matches `MatMul(x, W) -> Add(matmul_out, bias)` as an alternate
    encoding of Linear. Every prior experiment's exports produced a single
    `Gemm(transB=1)` node for `nn.Linear`; a REAL Hugging Face export
    (exp25/26) instead traces it as this 2-node pair, with `W` stored
    [in_dim, out_dim] — the mirror of Gemm's [out_dim, in_dim] convention,
    since a plain MatMul (no transpose flag) needs the weight
    pre-transposed to compute the same `x @ W_gemm^T`. The IR entry marks
    `weight_transposed=True` so weight-extraction transposes it back to
    [out,in] at compile time — the exact same generated C++ (`linear()`,
    common.hpp) handles both encodings unchanged; only which bytes land in
    the .bin file differs.
    Returns (ir_entry, next_index) or None."""
    if i + 1 >= len(nodes):
        return None
    n0, n1 = nodes[i], nodes[i + 1]
    if n0.op_type != "MatMul":
        return None
    w_candidates = [x for x in n0.input if x in initializer_names]
    x_candidates = [x for x in n0.input if x not in initializer_names]
    if len(w_candidates) != 1 or len(x_candidates) != 1:
        return None
    w_name, in_tensor = w_candidates[0], x_candidates[0]
    matmul_out = n0.output[0]

    if n1.op_type != "Add" or matmul_out not in n1.input:
        return None
    bias_candidates = [x for x in n1.input if x != matmul_out]
    if len(bias_candidates) != 1 or bias_candidates[0] not in initializer_names:
        return None
    b_name = bias_candidates[0]

    w_shape = initializers[w_name].shape
    if len(w_shape) != 2:
        return None
    in_dim, out_dim = w_shape
    if initializers[b_name].shape != (out_dim,):
        return None

    return {"op": "linear", "output": n1.output[0], "input": in_tensor,
            "weight": w_name, "bias": b_name, "in_dim": int(in_dim), "out_dim": int(out_dim),
            "weight_transposed": True}, i + 2


def _validate_causal_mask(nodes, constants, where_node):
    """Shared content-check for a `Trilu(ones,k=1) -> Cast -> Where(mask,
    -inf, scores)` triple, given the Where node — PyTorch's export of
    `scores.masked_fill(causal_mask, -inf)` for a standard strict
    upper-triangular (autoregressive) mask: row i attends to columns <= i
    only. Only that exact shape/fill-value/diagonal-offset is recognized;
    any other masking scheme falls through to this compiler's normal
    "unsupported op" rejection rather than being silently mishandled.
    Returns (pre_mask_tensor, consumed_indices_excluding_where) or None."""
    mask_in, true_in, false_in = where_node.input
    if true_in not in constants:
        return None
    fill_val = float(np.asarray(constants[true_in]).reshape(-1)[0])
    if not (math.isinf(fill_val) and fill_val < 0):
        return None

    cast_matches = [(idx, n) for idx, n in enumerate(nodes) if n.op_type == "Cast" and list(n.output) == [mask_in]]
    if len(cast_matches) != 1:
        return None
    cast_idx, cast_node = cast_matches[0]

    trilu_matches = [(idx, n) for idx, n in enumerate(nodes)
                      if n.op_type == "Trilu" and list(n.output) == [cast_node.input[0]]]
    if len(trilu_matches) != 1:
        return None
    trilu_idx, trilu_node = trilu_matches[0]
    if trilu_node.input[0] not in constants:
        return None
    ones_arr = np.asarray(constants[trilu_node.input[0]])
    if ones_arr.ndim < 2 or ones_arr.shape[-1] != ones_arr.shape[-2] or not np.all(ones_arr == 1):
        return None
    k_val = 1
    if len(trilu_node.input) > 1 and trilu_node.input[1] in constants:
        k_val = int(np.asarray(constants[trilu_node.input[1]]).reshape(-1)[0])
    if k_val != 1:
        return None  # only the standard strict-upper-triangular causal mask is recognized

    return false_in, {cast_idx, trilu_idx}


def try_match_causal_mask(nodes, constants, scores_tensor):
    """Forward variant: finds a causal mask applied TO scores_tensor (its
    unique Where-consumer, with scores_tensor as the false-branch). Used
    by try_match_self_attention and try_match_multihead_attention's
    Q/K-triggered branch, where the caller already has the pre-mask
    tensor and needs to find what (if anything) sits between it and
    softmax. Returns (masked_tensor, consumed_indices) or None — no mask
    present, scores_tensor feeds softmax directly (exp15-26's existing
    behavior, unchanged)."""
    where_matches = [(idx, n) for idx, n in enumerate(nodes) if n.op_type == "Where" and scores_tensor in n.input]
    if len(where_matches) != 1:
        return None
    where_idx, where_node = where_matches[0]
    if where_node.input[2] != scores_tensor:
        return None
    result = _validate_causal_mask(nodes, constants, where_node)
    if result is None:
        return None
    _, consumed = result
    return where_node.output[0], consumed | {where_idx}


def unwrap_causal_mask_backward(nodes, constants, tensor_name):
    """Backward variant: if tensor_name is itself a causal-mask Where
    node's output, unwraps to the tensor it masked. Used by
    try_match_multihead_attention's V-triggered branch, where the caller
    is tracing backward from softmax's input and needs to see past an
    optional mask to reach the pre-softmax MatMul. Returns
    (tensor_name_or_unwrapped, consumed_indices) — consumed_indices is
    empty when there's no mask to unwrap."""
    where_matches = [(idx, n) for idx, n in enumerate(nodes) if n.op_type == "Where" and list(n.output) == [tensor_name]]
    if len(where_matches) != 1:
        return tensor_name, set()
    where_idx, where_node = where_matches[0]
    result = _validate_causal_mask(nodes, constants, where_node)
    if result is None:
        return tensor_name, set()
    pre_mask_tensor, consumed = result
    return pre_mask_tensor, consumed | {where_idx}


def try_match_self_attention(nodes, i, constants, tensor_dims):
    """Looks for: Transpose(K,[1,0]) -> MatMul(Q,K^T) -> Mul/Div(scale) ->
    [optional causal mask] -> Softmax -> MatMul(attn,V), starting at
    nodes[i]. Returns (ir_entry, next_index) or None — a pattern match,
    not general MatMul/Transpose support; anything not shaped exactly
    like this falls through to the caller's normal per-node handling
    (which will reject Transpose/MatMul as unsupported op types on their
    own)."""
    if i + 2 >= len(nodes):
        return None
    n0, n1, n2 = nodes[i], nodes[i + 1], nodes[i + 2]

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

    mask_match = try_match_causal_mask(nodes, constants, scaled_tensor)
    causal = mask_match is not None
    softmax_input, consumed_mask = mask_match if causal else (scaled_tensor, set())

    softmax_matches = [(idx, n) for idx, n in enumerate(nodes)
                        if n.op_type == "Softmax" and list(n.input) == [softmax_input]]
    if len(softmax_matches) != 1:
        return None
    softmax_idx, softmax_node = softmax_matches[0]
    attn_tensor = softmax_node.output[0]

    mm2_matches = [(idx, n) for idx, n in enumerate(nodes) if n.op_type == "MatMul" and attn_tensor in n.input]
    if len(mm2_matches) != 1:
        return None
    mm2_idx, mm2_node = mm2_matches[0]
    v_candidates = [x for x in mm2_node.input if x != attn_tensor]
    if len(v_candidates) != 1:
        return None
    v_tensor = v_candidates[0]

    if q_tensor not in tensor_dims or k_tensor not in tensor_dims or v_tensor not in tensor_dims:
        return None
    d = tensor_dims[q_tensor]
    if tensor_dims[k_tensor] != d or tensor_dims[v_tensor] != d:
        return None

    consumed = {i, i + 1, i + 2, softmax_idx, mm2_idx} | consumed_mask
    ir_entry = {"op": "self_attention", "q": q_tensor, "k": k_tensor, "v": v_tensor,
                "scale": scale_value, "output": mm2_node.output[0], "dim": d, "causal": causal}
    return ir_entry, max(consumed) + 1


def try_match_multihead_attention(nodes, i, constants, tensor_dims, initializers):
    """Looks for the multi-head split/attend/merge pattern PyTorch's
    exporter emits for `q.view(seq,H,Dh).transpose(0,1)` etc: three
    (Reshape->Transpose) branches splitting Q/K/V into heads, a MatMul/
    scale/Softmax/MatMul attending per head (batched over the head dim),
    then a final Transpose->Reshape merging heads back. These nodes are
    NOT contiguous in nodes[i:] — PyTorch interleaves Q/K/V's linear nodes
    with the Reshape/Transpose nodes of branches parsed earlier — so this
    searches by tensor-name flow (who consumes whose output, who produces
    whose input) rather than position, starting only when nodes[i] looks
    like the first half of one head-split branch.

    exp23: every lookup here traces a SPECIFIC tensor name forward or
    backward from wherever nodes[i] sits (never "find all nodes shaped
    like X anywhere in the graph") — necessary once a graph can contain
    more than one attention instance (stacked encoder blocks).

    exp26: generalized twice more, both times because a REAL Hugging Face
    export (exp25) shapes this differently than every synthetic test model
    had: (1) accepts either the 3D convention (batch squeezed out —
    reshape to [seq,H,Dh], perm [1,0,2]/[1,2,0]) or the 4D convention
    (explicit batch=1 — reshape to [1,seq,H,Dh], perm [0,2,1,3]/[0,2,3,1])
    via classify_perm(); (2) accepts scaling applied as ONE Mul/Div after
    the pre-softmax MatMul (exp20/22's synthetic models) OR as TWO
    separate Mul/Div ops applied to Q and K individually BEFORE that
    MatMul (HF's actual SDPA-derived export, each by d_head**-0.25) via
    unwrap_scale_forward/backward, which walk past zero-or-one such
    scale-wrap nodes in either direction and fold the factor into one
    combined scale for codegen either way.

    Returns (ir_entry, consumed_indices) or None. consumed_indices is the
    full set of node indices this match uses; the caller must defer
    appending ir_entry to the IR list until it reaches max(consumed_indices)
    — Q/K/V's own linear nodes (not part of consumed_indices) may sit at
    later positions than nodes[i], and the generated C++ must declare their
    outputs before the fused op reads them."""
    node0 = nodes[i]
    if node0.op_type != "Reshape" or node0.input[1] not in constants:
        return None
    shape0 = [int(x) for x in np.asarray(constants[node0.input[1]]).reshape(-1)]
    if len(shape0) == 4 and shape0[0] != 1:
        return None
    if len(shape0) not in (3, 4):
        return None
    d_head = shape0[-1]

    def resolve_dim(tensor_name):
        """A tensor's feature dim — from tensor_dims if the main walk has
        already processed its producer, else by looking directly at
        whichever node produces it (needed here because, e.g., K's and
        V's own linear nodes may sit AFTER nodes[i] in the node list, not
        yet reached by the main walk when Q's branch triggers this match
        first). Reads the bias's own 1-D shape rather than the weight's,
        since that's the same regardless of whether the linear is a single
        Gemm(transB=1) (exp15+) or a MatMul+Add pair (exp26 — modern
        torch.onnx emits this instead for some real models' Linear)."""
        if tensor_name in tensor_dims:
            return tensor_dims[tensor_name]
        for n in nodes:
            if list(n.output) != [tensor_name]:
                continue
            if n.op_type in ("Gemm", "Add"):
                bias_candidates = [x for x in n.input if x in initializers and initializers[x].ndim == 1]
                if len(bias_candidates) == 1:
                    return int(initializers[bias_candidates[0]].shape[0])
        return None

    d = resolve_dim(node0.input[0])
    if d is None or d_head <= 0 or d % d_head != 0:
        return None
    num_heads = d // d_head

    def find_consumers(tensor_name, op_type):
        return [(idx, n) for idx, n in enumerate(nodes) if tensor_name in n.input and n.op_type == op_type]

    def find_producer(tensor_name):
        """Unfiltered — returns whichever node produces this tensor,
        regardless of op_type, so the caller can branch on what kind of
        node it turns out to be. Every lookup in this function is
        tensor-name-exact, so it never widens to nodes from a different
        attention instance (exp23) or misreads a differently-shaped real
        export (exp26) as something it isn't."""
        matches = [(idx, n) for idx, n in enumerate(nodes) if list(n.output) == [tensor_name]]
        return matches[0] if len(matches) == 1 else None

    def perm_of(transpose_node):
        return next((list(a.ints) for a in transpose_node.attribute if a.name == "perm"), None)

    def classify_perm(perm):
        """'qv' or 'k' for either the 3D convention (perm [1,0,2]/[1,2,0])
        or the 4D batch=1 convention (perm [0,2,1,3]/[0,2,3,1] — same
        split, PyTorch just carries the batch dim through). None if it
        matches neither."""
        if perm in ([1, 0, 2], [0, 2, 1, 3]):
            return "qv"
        if perm in ([1, 2, 0], [0, 2, 3, 1]):
            return "k"
        return None

    def unwrap_scale_forward(tensor_name):
        """If tensor_name's unique consumer is a Mul/Div-by-constant,
        follows through it. Returns (final_tensor, cumulative_scale,
        consumed_idx_or_None)."""
        consumers = [(idx, n) for idx, n in enumerate(nodes) if tensor_name in n.input]
        if len(consumers) != 1 or consumers[0][1].op_type not in ("Mul", "Div"):
            return tensor_name, 1.0, None
        idx, node = consumers[0]
        match = try_match_scale(node, tensor_name, constants)
        if match is None:
            return tensor_name, 1.0, None
        scale, out_tensor = match
        return out_tensor, scale, idx

    def unwrap_scale_backward(tensor_name):
        """Mirror of unwrap_scale_forward, walking backward: if
        tensor_name's producer is a Mul/Div-by-constant, unwraps to the
        tensor it scaled. Returns (underlying_tensor, cumulative_scale,
        (producer_idx, producer_node), consumed_idx_or_None) — producer is
        whichever REAL (non-scale) op produced the unwrapped tensor."""
        producer = find_producer(tensor_name)
        if producer is None:
            return None
        idx, node = producer
        if node.op_type in ("Mul", "Div"):
            non_const = [x for x in node.input if x not in constants]
            if len(non_const) == 1:
                match = try_match_scale(node, non_const[0], constants)
                if match is not None:
                    scale, _ = match
                    return non_const[0], scale, None, idx  # caller re-resolves producer of non_const[0]
        return tensor_name, 1.0, (idx, node), None

    def resolve_backward(tensor_name):
        """Repeatedly applies unwrap_scale_backward until it reaches a
        real (non-scale) producer. Returns (final_tensor, total_scale,
        (producer_idx, producer_node), consumed_indices_set)."""
        total_scale = 1.0
        consumed_here = set()
        while True:
            result = unwrap_scale_backward(tensor_name)
            if result is None:
                return None
            tensor_or_final, scale, producer, skipped_idx = result
            total_scale *= scale
            if skipped_idx is not None:
                consumed_here.add(skipped_idx)
                tensor_name = tensor_or_final
                continue
            return tensor_or_final, total_scale, producer, consumed_here

    def branch_from_reshape(r_idx, r_node):
        """(Reshape, Transpose) pair starting at r_node — verifies shape/dim
        match and returns the branch dict, or None."""
        if r_node.input[1] not in constants:
            return None
        shape = [int(x) for x in np.asarray(constants[r_node.input[1]]).reshape(-1)]
        if shape != shape0 or resolve_dim(r_node.input[0]) != d:
            return None
        consumers = find_consumers(r_node.output[0], "Transpose")
        if len(consumers) != 1:
            return None
        t_idx, t_node = consumers[0]
        role = classify_perm(perm_of(t_node) or [])
        if role is None:
            return None
        return {"reshape_idx": r_idx, "transpose_idx": t_idx, "role": role,
                "out": t_node.output[0], "linear": r_node.input[0]}

    def branch_from_transposed_tensor(tensor_name):
        """Traces backward from a (possibly pre-scaled) Transpose output
        tensor to its (Reshape, Transpose) branch. Returns
        (branch_dict, scale_factor, consumed_indices) or None."""
        resolved = resolve_backward(tensor_name)
        if resolved is None:
            return None
        real_tensor, scale, producer, consumed_here = resolved
        if producer[1].op_type != "Transpose":
            return None
        t_idx, t_node = producer
        r_producer = find_producer(t_node.input[0])
        if r_producer is None or r_producer[1].op_type != "Reshape":
            return None
        r_idx, r_node = r_producer
        branch = branch_from_reshape(r_idx, r_node)
        if branch is None or branch["transpose_idx"] != t_idx:
            return None
        return branch, scale, consumed_here

    branch0 = branch_from_reshape(i, node0)
    if branch0 is None:
        return None

    consumed_scale_nodes = set()

    mm_a_tensor, prescale0, prescale0_idx = unwrap_scale_forward(branch0["out"])
    if prescale0_idx is not None:
        consumed_scale_nodes.add(prescale0_idx)
    mm_a_matches = find_consumers(mm_a_tensor, "MatMul")
    if len(mm_a_matches) != 1:
        return None
    mm_a_idx, mm_a_node = mm_a_matches[0]
    other = [x for x in mm_a_node.input if x != mm_a_tensor]
    if len(other) != 1:
        return None
    other_resolved = resolve_backward(other[0])
    if other_resolved is None:
        return None
    _, _, other_producer, other_consumed = other_resolved
    consumed_scale_nodes |= other_consumed

    if other_producer[1].op_type == "Softmax":
        # branch0 is V (feeds the post-softmax MatMul); trace backward
        # through softmax -> mm_pre to find Q and K.
        if branch0["role"] != "qv":
            return None
        v_branch, v_scale = branch0, prescale0
        mm_post_idx, mm_post_node = mm_a_idx, mm_a_node
        softmax_idx, softmax_node = other_producer

        pre_softmax_tensor, mask_consumed = unwrap_causal_mask_backward(nodes, constants, softmax_node.input[0])
        causal = bool(mask_consumed)
        consumed_scale_nodes |= mask_consumed

        mm_pre_result = resolve_backward(pre_softmax_tensor)
        if mm_pre_result is None:
            return None
        mm_pre_tensor, mm_pre_scale, mm_pre_producer, mm_pre_consumed = mm_pre_result
        if mm_pre_producer[1].op_type != "MatMul":
            return None
        consumed_scale_nodes |= mm_pre_consumed
        mm_pre_idx, mm_pre_node = mm_pre_producer
        qk_tensors = list(mm_pre_node.input)
        if len(qk_tensors) != 2:
            return None
        result_a = branch_from_transposed_tensor(qk_tensors[0])
        result_b = branch_from_transposed_tensor(qk_tensors[1])
        if result_a is None or result_b is None:
            return None
        branch_a, scale_a, consumed_a = result_a
        branch_b, scale_b, consumed_b = result_b
        consumed_scale_nodes |= consumed_a | consumed_b
        if branch_a["role"] == "k" and branch_b["role"] == "qv":
            k_branch, q_branch, qk_scale = branch_a, branch_b, scale_a * scale_b
        elif branch_a["role"] == "qv" and branch_b["role"] == "k":
            q_branch, k_branch, qk_scale = branch_a, branch_b, scale_a * scale_b
        else:
            return None
        scale_value = mm_pre_scale * qk_scale * v_scale

    elif other_producer[1].op_type == "Transpose":
        # branch0 is Q or K (both feed mm_pre, the pre-softmax MatMul);
        # trace forward through softmax -> mm_post to find V.
        other_result = branch_from_transposed_tensor(other[0])
        if other_result is None:
            return None
        other_branch, other_scale, other_branch_consumed = other_result
        consumed_scale_nodes |= other_branch_consumed
        mm_pre_idx, mm_pre_node = mm_a_idx, mm_a_node
        if branch0["role"] == "k" and other_branch["role"] == "qv":
            k_branch, q_branch = branch0, other_branch
        elif branch0["role"] == "qv" and other_branch["role"] == "k":
            q_branch, k_branch = branch0, other_branch
        else:
            return None
        qk_scale = prescale0 * other_scale

        post_scale_tensor, post_scale, post_scale_idx = unwrap_scale_forward(mm_pre_node.output[0])
        if post_scale_idx is not None:
            consumed_scale_nodes.add(post_scale_idx)

        mask_match = try_match_causal_mask(nodes, constants, post_scale_tensor)
        causal = mask_match is not None
        softmax_input, mask_consumed = mask_match if causal else (post_scale_tensor, set())
        consumed_scale_nodes |= mask_consumed

        softmax_matches = find_consumers(softmax_input, "Softmax")
        if len(softmax_matches) != 1:
            return None
        softmax_idx, softmax_node = softmax_matches[0]

        mm_post_matches = find_consumers(softmax_node.output[0], "MatMul")
        if len(mm_post_matches) != 1:
            return None
        mm_post_idx, mm_post_node = mm_post_matches[0]
        v_candidates = [x for x in mm_post_node.input if x != softmax_node.output[0]]
        if len(v_candidates) != 1:
            return None
        v_result = branch_from_transposed_tensor(v_candidates[0])
        if v_result is None:
            return None
        v_branch, v_scale, v_consumed = v_result
        if v_branch["role"] != "qv":
            return None
        consumed_scale_nodes |= v_consumed
        scale_value = qk_scale * post_scale * v_scale
    else:
        return None

    final_t_matches = find_consumers(mm_post_node.output[0], "Transpose")
    if len(final_t_matches) != 1:
        return None
    final_t_idx, final_t_node = final_t_matches[0]
    if classify_perm(perm_of(final_t_node) or []) != "qv":
        return None

    final_r_matches = find_consumers(final_t_node.output[0], "Reshape")
    if len(final_r_matches) != 1:
        return None
    final_r_idx, final_r_node = final_r_matches[0]
    if final_r_node.input[1] not in constants:
        return None
    final_shape = [int(x) for x in np.asarray(constants[final_r_node.input[1]]).reshape(-1)]
    if len(final_shape) not in (2, 3) or final_shape[-1] not in (-1, d):
        return None
    if len(final_shape) == 3 and final_shape[0] not in (-1, 1):
        return None

    branches = (q_branch, k_branch, v_branch)
    consumed = {b["reshape_idx"] for b in branches} | {b["transpose_idx"] for b in branches} | {
        mm_pre_idx, softmax_idx, mm_post_idx, final_t_idx, final_r_idx,
    } | consumed_scale_nodes
    ir_entry = {"op": "multihead_self_attention", "q": q_branch["linear"], "k": k_branch["linear"],
                "v": v_branch["linear"], "num_heads": num_heads, "d_head": d_head,
                "scale": scale_value, "output": final_r_node.output[0], "dim": d, "causal": causal}
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

        gelu_match = try_match_decomposed_gelu(nodes, i, constants, tensor_dims)
        if gelu_match is not None:
            ir_entry, next_i = gelu_match
            tensor_dims[ir_entry["output"]] = ir_entry["dim"]
            ir.append(ir_entry)
            i = next_i
            continue

        tanh_gelu_match = try_match_tanh_gelu(nodes, i, constants, tensor_dims)
        if tanh_gelu_match is not None:
            ir_entry, next_i = tanh_gelu_match
            tensor_dims[ir_entry["output"]] = ir_entry["dim"]
            ir.append(ir_entry)
            i = next_i
            continue

        linear_match = try_match_matmul_add_linear(nodes, i, initializer_names, initializers)
        if linear_match is not None:
            ir_entry, next_i = linear_match
            tensor_dims[ir_entry["input"]] = ir_entry["in_dim"]
            tensor_dims[ir_entry["output"]] = ir_entry["out_dim"]
            ir.append(ir_entry)
            i = next_i
            continue

        identity_match = try_match_identity_reshape(node, constants, tensor_dims)
        if identity_match is not None:
            tensor_dims[identity_match["output"]] = identity_match["dim"]
            ir.append(identity_match)
            i += 1
            continue

        if node.op_type not in SUPPORTED_OPS:
            raise UnsupportedGraph(
                f"op '{node.op_type}' is not supported by this compiler outside the recognized "
                f"self-attention / multi-head-attention / decomposed-GELU / tanh-GELU / "
                f"MatMul+Add-linear / batch-squeeze-reshape patterns "
                f"(supported standalone: {sorted(SUPPORTED_OPS)})"
            )
        if len(node.output) != 1:
            raise UnsupportedGraph(f"op '{node.op_type}' has multiple outputs, not supported")

        non_init_inputs = [x for x in node.input if x not in initializer_names]
        out_name = node.output[0]
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}

        if node.op_type == "Gemm":
            if attrs.get("alpha", 1.0) != 1.0 or attrs.get("beta", 1.0) != 1.0:
                raise UnsupportedGraph(f"Gemm with non-standard attrs (need alpha=beta=1): {attrs}")
            trans_b = attrs.get("transB", 0)
            if trans_b not in (0, 1):
                raise UnsupportedGraph(f"Gemm with non-standard attrs (transB must be 0 or 1): {attrs}")
            if len(non_init_inputs) != 1:
                raise UnsupportedGraph(f"Gemm with {len(non_init_inputs)} tensor inputs, expected 1")
            in_tensor = non_init_inputs[0]
            _, w_name, b_name = node.input
            # transB=1 (the usual nn.Linear export): weight stored [out,in],
            # matching linear()'s (common.hpp) native layout directly.
            # transB=0 (exp28: GPT-2's Conv1D, `addmm(bias, x, weight)` with
            # no transpose): weight stored [in,out], the same "mirror"
            # convention exp26's MatMul+Add already handles — transposed
            # back to [out,in] at weight-extraction time, same flag reused.
            if trans_b == 1:
                out_dim, in_dim = initializers[w_name].shape
            else:
                in_dim, out_dim = initializers[w_name].shape
            tensor_dims[in_tensor] = int(in_dim)
            tensor_dims[out_name] = int(out_dim)
            ir_entry = {"op": "linear", "output": out_name, "input": in_tensor,
                        "weight": w_name, "bias": b_name, "in_dim": int(in_dim), "out_dim": int(out_dim)}
            if trans_b == 0:
                ir_entry["weight_transposed"] = True
            ir.append(ir_entry)

        elif node.op_type in ("Relu", "Gelu", "Softmax"):
            if len(non_init_inputs) != 1:
                raise UnsupportedGraph(f"{node.op_type} with {len(non_init_inputs)} tensor inputs, expected 1")
            in_tensor = non_init_inputs[0]
            op_name = {"Relu": "relu", "Gelu": "gelu", "Softmax": "softmax"}[node.op_type]
            if node.op_type == "Gelu":
                # exp28: opset>=20's fused Gelu carries an `approximate`
                # attribute — "tanh" (GPT-2's "gelu_new") is a genuinely
                # DIFFERENT function from the default exact/erf formula,
                # not just a different encoding of the same one (unlike
                # exp24's decomposed-vs-fused exact GELU), so it gets its
                # own IR op and codegen, not the existing "gelu"'s.
                approx = attrs.get("approximate", b"none")
                if approx in (b"tanh", "tanh"):
                    op_name = "gelu_tanh"
                elif approx not in (b"none", "none"):
                    raise UnsupportedGraph(f"Gelu approximate={approx!r} not supported — only exact/erf or tanh")
            if node.op_type == "Softmax" and attrs.get("axis", -1) not in (-1, 1):
                raise UnsupportedGraph(f"Softmax over unsupported axis: {attrs.get('axis')}")
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
            # exp28: pre-norm models (LayerNorm before the sub-block, not
            # after) can make LayerNorm the FIRST consumer of the graph's
            # own input tensor — every prior model was post-norm, where
            # some earlier Linear had already recorded the input's dim.
            tensor_dims[in_tensor] = dim
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

    if not ir:
        raise UnsupportedGraph("this compiler produced an empty graph")

    return ir, initializers, graph_input_name


def generate_cpp(ir, graph_input_name):
    """One block of C++ per IR op, reading/writing named variables that
    correspond directly to ONNX tensor names."""

    def var(tensor_name):
        return "X" if tensor_name == graph_input_name else cpp_id(tensor_name)

    # exp28: a pre-norm model (LayerNorm before the sub-block, not after —
    # GPT-2's convention) can start with `layernorm` rather than `linear`;
    # every dim-preserving op ("dim") or Linear ("in_dim") knows its own
    # input width either way.
    input_dim = ir[0].get("in_dim", ir[0].get("dim"))
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
        elif op["op"] == "identity":
            # A pure batch=1 squeeze/unsqueeze reshape (exp28) — nothing to
            # compute, just a name for the same [n, dim] data.
            in_var = var(op["input"])
            forward_lines.append(f"        std::vector<float> {out_var} = {in_var};")
        elif op["op"] in ("relu", "gelu", "gelu_tanh", "softmax"):
            in_var = var(op["input"])
            forward_lines.append(f"        std::vector<float> {out_var} = {in_var};")
            if op["op"] == "softmax":
                forward_lines.append(f"        softmax_rows({out_var}.data(), n, {op['dim']});")
            else:
                fn = {"relu": "relu_inplace", "gelu": "gelu_inplace", "gelu_tanh": "gelu_tanh_inplace"}[op["op"]]
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
            if op.get("causal"):
                forward_lines.append(f"        causal_mask_rows({scores_var}.data(), n);")
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
            if op.get("causal"):
                forward_lines.append(f"            causal_mask_rows({scores_var}.data(), n);")
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
            weight = initializers[op["weight"]]
            if op.get("weight_transposed"):
                # MatMul+Add's W is stored [in,out] (exp26) — linear()
                # (common.hpp) expects [out,in], same as Gemm(transB=1)
                # already provides; .T is a view, but tofile() always
                # writes in C order of the (transposed) shape, so this
                # lands correctly without an explicit copy.
                weight = weight.T
            weight.astype(np.float32).tofile(out_dir / f"{cpp_id(op['weight'])}.bin")
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
