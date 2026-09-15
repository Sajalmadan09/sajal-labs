# Experiment 15 — The Smallest Real Compiler: ONNX to Native C++

**Why this experiment**: every prior experiment (exp1-14) required a new hand-written C++ header per architecture (`mlp_model.hpp`, `transformer_model.hpp`, `bert_model.hpp`). That validates the *hypothesis* (native artifacts can preserve fidelity and win on cold-invocation), but it doesn't validate the *project's actual claim* — that trained models can be *compiled*, not hand-ported, into native artifacts. `research/gap-analysis.md`'s revised recommendation already concluded Sajal Labs is primarily a compiler; this is the first experiment that actually builds one, even a minimal one.

**Deliberately narrow scope, stated upfront**: this compiler supports exactly one architecture class — `Gemm → Relu → Gemm → Softmax` (i.e. `Linear → ReLU → Linear → Softmax`, exp1 and exp9's shape) — three ONNX op types, no fusion, no graph optimization, no dynamic shapes, no control flow. The question this experiment answers is **"can auto-generated code be correct at all,"** not "can we compile arbitrary models" or "is the generated code fast" — those are separate, later questions, deliberately not conflated with this one (see exp2-8's own lesson about isolating one variable at a time).

## What was built

[python/sajal_compile.py](../../../python/sajal_compile.py):
1. **Parser**: loads an ONNX graph (`onnx.load`), walks nodes, recognizes exactly `Gemm` (with `transB=1`, matching PyTorch's `nn.Linear` export convention — verified against the real graph structure of both exp1's and exp9's `model.onnx` before writing any parsing code, not assumed), `Relu`, `Softmax`. Anything else — or this exact sequence in a different order — raises `UnsupportedGraph` with a specific, actionable message rather than guessing.
2. **IR**: a flat list of `{op, dims, weight/bias initializer names}` dicts — no separate optimization pass, since there's nothing to optimize at this scope.
3. **Codegen**: emits one line of C++ per IR op, each a call into `common.hpp`'s already-validated shared ops (`linear`, `relu_inplace`, `softmax_rows`) — the compiler never generates a numerical kernel itself, only wires together kernels that were already proven correct across exp1-14.
4. **Output convention**: weights extracted to the exact same `fc1_weight.bin`/`shapes.txt` layout `mlp_model.hpp` already expects — a deliberate choice, not an accident, so every existing tool works on compiler output with zero modification (see Validation below).

## Validation — four independent checks, not one

**1. Equivalence via the existing, unmodified `compare.py`** (same PyTorch reference every prior experiment used, copied through from the source artifacts):

| Model | Max abs error | Matches exp1/exp9's original number? |
|---|---:|---|
| exp1's MLP (compiled from `artifacts/model.onnx`) | 4.47e-08 | Yes — identical to exp1's original result |
| exp9's gender classifier (compiled from `artifacts/gender_classifier/model.onnx`) | 1.19e-07 | Yes — identical to exp9's original result |

**2. Byte-for-byte comparison against the hand-written binary** — not just "close," checked directly: `cmp compiled_output.bin handwritten_output.bin` → **bit-identical** for both models. The auto-generated code doesn't approximate the hand-written reference; for this architecture class, it performs the exact same sequence of floating-point operations, in the exact same order, down to the last bit.

**3. The `sajal` CLI works on compiler output with zero modification** — `sajal inspect`/`run`/`bench` against `artifacts/compiled/exp1_mlp/` correctly detect "mlp" kind, report the right dims (64/128/10, extracted from the ONNX graph, not hand-typed), and run correctly. `sajal.cpp` has no knowledge this experiment exists; it just sees a directory matching a convention it already understood.

**4. Negative-path test — does it fail loudly on graphs outside its scope, or guess?** Ran the compiler against exp2's transformer ONNX and exp11's real BERT ONNX (both genuinely outside this compiler's supported class):

```
error: op 'Identity' is not supported by this minimal compiler (supported: Gemm, Relu, Softmax)
error: op 'Shape' is not supported by this minimal compiler (supported: Gemm, Relu, Softmax)
```

Both fail cleanly and immediately, before generating or compiling anything — no silent wrong output, no best-effort guess at an unsupported op.

## The generated code

[generated_exp1_mlp.cpp](generated_exp1_mlp.cpp) — the actual output for exp1's model, unedited. Worth reading directly rather than taking the equivalence numbers on faith: it's genuinely generated (one line per IR op) and happens to be structurally almost identical to the hand-written `mlp_model.hpp` — which is itself a form of validation. An independently-derived compiler arriving at essentially the same C++ a human wrote by hand is stronger evidence of correctness than either alone.

## Interpretation

This is a real, if narrow, answer to RQ1/RQ8 from the original brief ("can trained models be compiled, not hand-ported, into native artifacts") — yes, for this architecture class, with bit-identical fidelity to both the original PyTorch model and the hand-written native port. The scope is deliberately tiny: three op types, one fixed architecture shape, no optimization. That's the right size for a first compiler experiment, not a limitation to apologize for — every prior experiment in this project succeeded by asking one falsifiable question at a time, and "does codegen produce correct output at all" is a different, prerequisite question from "is codegen's output fast" or "how many architectures can it cover."

## Caveats

Three op types only (`Gemm`, `Relu`, `Softmax`) — no `LayerNorm`, `GELU`, `Embedding`, or attention, so BERT-tiny and the transformer block are explicitly out of scope for now, not silently mishandled (confirmed by the negative-path test). No graph optimization or operator fusion — codegen is a direct 1:1 IR-to-call translation, so this experiment says nothing about whether compiled code can be competitively fast, only that it can be correct. No dynamic shapes or control flow. Fixed architecture *sequence* (exactly `linear→relu→linear→softmax`) rather than an arbitrary DAG — a real compiler eventually needs graph traversal more general than a fixed pattern match, but that generality wasn't needed to answer this experiment's question and would have been premature scope.

## Next experiment

Two natural, separate directions, deliberately not bundled into one: (1) widen op coverage incrementally — `LayerNorm` and `GELU` next (needed for the transformer block, exp2-8's shape), with the same bit-identical-vs-hand-written validation bar before adding attention/`Embedding` for BERT; (2) once correctness is established for a wider op set, *then* ask the fusion/optimization question this experiment deliberately deferred — does compiled-but-unfused code lose the warm-latency edge the same way exp5's hand-written multi-call attention did, and can the compiler's codegen be taught to fuse the way a human optimizing by hand would (exp5's actual finding about GEMM call count)?
