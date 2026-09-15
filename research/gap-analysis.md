# Gap Analysis & Phase 0 Recommendation

Synthesis of [landscape.md](landscape.md) (CPU/edge runtimes), [related-work.md](related-work.md) (LLM-serving systems + compilers/math libs), and [papers.md](papers.md) (academic literature). This is judgment, not another research pass — the conclusions below are mine, weighing what the four surveys found against each other.

> **Update after experiments 1-7 (2026-09-16): the value proposition below has been revised.** This doc originally (Phase 0, before any code existed) bet on native C++ being competitive on *warm per-call latency* for small/medium models, treating cold-start/footprint as a secondary benefit. Seven experiments later, that bet didn't pay off: native never beat ONNX Runtime CPU on warm latency for any transformer-shaped workload (exp2-6), even after diagnosing and testing two specific hypotheses (buffer allocation, BLAS dispatch overhead — exp4/exp5) and one direct fix attempt that made things worse (exp6). What held up in *every* experiment, consistently, regardless of model architecture: **cold-start (15-155x) and dependency-footprint (~10-40x) advantages** — see [exp7](experiments/exp7-cold-invocation/results.md). The project's value proposition is now **single-shot/cold-invocation latency and dependency footprint, not warm-server throughput** — this changes which of the niches below is actually load-bearing (see the revised MVP recommendation at the end of this document, added after the original).

## Where is the actual gap?

Going through the candidate differentiators from the brief, checked against what the research actually found:

**A. Model-specific native compilation** — *not novel by itself.* ExecuTorch's selective build (links only the operators a specific `.pte` uses, ~50KB core), ONNX Runtime's reduced-operator-config custom build, and MNN's `MNN_BUILD_MINI` all already do "only include what this model needs." TensorRT goes further — it's a true per-model AOT compiler producing a hardware-specific fused-kernel engine. Claiming this alone as the differentiator would be claiming something three mature projects already ship.

**B. Minimal runtime (only required operators)** — *same as A, already solved.*

**C. Model + runtime single-artifact distribution** — *partially open.* GGUF/llama.cpp is the only project surveyed that ships one self-contained file with zero separate runtime to link — but it's architecturally LLM-shaped (tokenizer, KV-cache, attention kernels baked into ggml). ExecuTorch's `.pte` is a single artifact but still requires linking the separate ExecuTorch runtime. **Nobody does GGUF-style single-file distribution for general small/medium neural nets** (classifiers, embedding models, small transformers). This is real, but narrow.

**D. Model + runtime distribution via Hugging Face** — *nascent, not owned.* `huggingface/optimum-executorch` already does Hub → `.pte` export for a fixed list of architectures (Llama, Gemma, Qwen, OLMo). Prior art exists; it's narrow (fixed architecture list) and worth differentiating against explicitly, not ignoring.

**E. Extremely simple developer UX** (`sajal pull`, `sajal serve`) — *not a technical differentiator.* Every project surveyed already has some CLI/library UX; simplicity of API design is good practice, not a research contribution.

**F. Behavior-preserving conversion / equivalence testing** — ***this is the real gap.*** Per papers.md: none of the fusion/compiler literature (DNNFusion, TASO, TVM, XLA) treats "does the optimized output match the unoptimized reference to a stated tolerance" as a first-class, measured property — they validate task accuracy or algebraic equivalence, not floating-point numerical equivalence. The two closest empirical studies (Openja et al. 2022, Louloudakis et al. 2023) measure conversion fidelity between full frameworks/runtimes (ONNX, CoreML) — not compilation down to a dependency-free C artifact with hand-written kernels, where basic runtime services (dynamic shapes, broadcasting, libm rounding) may not even be available. **No dedicated study exists at exactly this intersection.**

**G. Tiny/medium models get less attention than LLMs** — *true, and the second real gap.* Every GPU-serving system surveyed (TensorRT-LLM, vLLM, SGLang, LMDeploy) is built around autoregressive decoding and KV-cache management — irrelevant machinery for a classifier or embedding model with a single forward pass. The closest existing prior art for "compile one small model into minimal native code" is **m2cgen** (classical ML — linear models, tree ensembles — to C/Java/Go source) and **MCUNet's TinyEngine** (model-adaptive memory scheduling/kernel specialization, but for a fixed NAS-derived architecture on microcontrollers). Neither covers "arbitrary small/medium *neural* models (including small transformers) compiled to native C with measured numerical fidelity."

### The actual niche

The defensible intersection is narrower than "native inference" and narrower than "minimal runtime" — both are taken. It is:

> **Per-model AOT compilation of small/medium neural networks (not LLMs, not classical ML) into dependency-free C/C++ artifacts, with numerical-equivalence measurement against the original framework as a first-class, published methodology — not an afterthought.**

This combines gap F (equivalence measurement, genuinely under-studied) with gap G (small/medium neural models, underserved relative to LLM-serving attention) using the compilation *style* already validated as the right architecture by MCUNet, ExecuTorch, and TVM (graph IR → fusion → codegen), but targeting a artifact shape closer to m2cgen/GGUF (self-contained, no separate runtime SDK to link) than to ONNX Runtime/TFLite (link a general interpreter library).

## What Sajal Labs should be

Per the brief's options (A–F): **a combination, weighted toward compiler + benchmark/research project, explicitly NOT a new runtime or new model format for now.**

- **Not (A) a general runtime** — ONNX Runtime, TFLite/LiteRT, ncnn, MNN, XNNPACK already own this. Building a competing general operator-dispatch interpreter has no justification.
- **Yes, primarily (B) a compiler** — but scoped to per-model AOT C/C++ codegen for small/medium neural nets, mirroring the graph-IR → fusion → codegen architecture that TVM/ExecuTorch/MCUNet already validated, not a new multi-backend compiler stack (that's TVM/MLIR/IREE's job — don't rebuild it).
- **Not (C) a new model format**, yet — reuse safetensors/ONNX/PyTorch state_dict as *input*; the *output* is generated C/C++ source + weights, not a new binary spec to design and maintain. GGUF-style single-file packaging is a plausible later step, not a Phase-0/1 decision.
- **Not (D) a distribution ecosystem** as the starting point — Hugging Face publishing is a Phase 7 goal (per the brief's own roadmap), downstream of having something real to publish.
- **Yes, essentially (E) a benchmark/research project** — the equivalence-measurement methodology (gap F above) is itself a publishable contribution, independent of whether the compiler ever becomes production-grade. This is not a fallback if the compiler "doesn't pan out" — it's a first-class deliverable either way, because it's the piece the existing literature is genuinely missing.

## Recommended smallest meaningful MVP

Collapse Phases 1–4 of the brief's roadmap into one small, honest, real first experiment, before committing to anything bigger:

1. **One small neural net** (`Linear → ReLU → Linear → Softmax`, per the brief's own §21/§22), trained in PyTorch.
2. **Hand-written C/C++ implementation** of the same forward pass, calling Apple's Accelerate framework for the matmuls (per related-work.md — the right first backend on this M4 hardware, zero-install, AMX-backed) with Eigen as header-only fallback for anything Accelerate doesn't cover.
3. **A rigorous equivalence harness**: same test inputs through PyTorch and the C++ version, reporting max absolute error, RMSE, cosine similarity, and prediction agreement — with an explicit, stated tolerance policy (not a single blanket epsilon), citing Goldberg 1991 and Shanmugavelu et al. 2024 for *why* exact equality shouldn't be expected.
4. **An honest benchmark** on this machine only (Apple M4, no CUDA claims): cold start, warm-up, P50/P95/P99 latency, peak RSS, binary size — for the C++ version vs. PyTorch eager vs. ONNX Runtime CPU EP, at batch size 1, single-threaded, with hardware/compiler/flags documented per the brief's §12 fairness rules.

This is small enough to finish and be honest about, and it directly tests the project's own central hypothesis (RQ2, RQ3, RQ5) before any compiler-generality work is justified. If the C++ version isn't meaningfully faster/leaner than ONNX Runtime CPU for this trivial model, that's a real, publishable finding too — not a failure to hide.

**Do not start with a transformer or an existing pretrained model.** Per the brief's own §7 and this research: the tiny hand-built net isolates "does a from-scratch C++ implementation help at all" from "did we correctly port someone else's architecture," which is exactly the confound §9 warns against.

## What to explicitly avoid building (per this research)

- Hand-written SIMD GEMM/conv kernels for CPU (link Accelerate/Eigen/OpenBLAS instead — landscape.md, related-work.md).
- A new quantization scheme (universally solved — landscape.md cross-cutting, papers.md §1).
- A new ML compiler IR/auto-scheduler (TVM/MLIR/IREE already solved this — related-work.md).
- A new LLM runtime or GGUF-alternative format (llama.cpp owns this — related-work.md).
- A production multi-tenant HTTP serving platform (Clipper/TensorFlow-Serving/TorchServe/OVMS territory — papers.md §6, landscape.md) — worth revisiting only after the compiler itself is validated.

---

## Revised recommendation (post-experiments, 2026-09-16): cold-invocation, not warm-throughput

The gap identified above (§F+G, "per-model AOT compilation with measured numerical equivalence, for small/medium neural nets") is still the right technical niche — nothing in exp1-7 changed the ecosystem survey's conclusions. What changed is **which benefit of that niche is actually the load-bearing one.**

**What the experiments showed:**
- exp1 (trivial 2-matmul MLP): native won warm latency 6-13x. Looked like validation of the original "native beats framework dispatch overhead" thesis.
- exp2 (transformer block, ~2,700x more compute): that warm-latency win nearly vanished (~1.0x, statistical tie with ONNX Runtime).
- exp3 (size sweep, `d_model` 16-256): the tie wasn't about compute scale at all — even the *smallest* transformer tested was already at parity with ONNX Runtime, contradicting the "compute-bound vs. dispatch-bound crossover" theory from exp2.
- exp4 (buffer-reuse fix): ruled out per-call heap allocation as the cause.
- exp5 (GEMM dispatch microbenchmark): found the real mechanism — Accelerate's fixed per-`cblas_sgemm`-call cost dominates at attention-head matrix sizes, confirmed in isolation (up to 3.77x penalty for more calls at equal total work).
- exp6 (hand-written attention, avoiding BLAS calls entirely): made things *worse* at every size — Accelerate's kernel efficiency (AMX/SIMD) outweighs its own dispatch overhead once you're not calling it at all.
- exp7 (true cold-invocation latency, external wall-clock, both architectures): **15-155x faster than Python across both the MLP and the transformer**, a result that is consistent across architecture in a way none of the warm-latency numbers ever were.

**The conclusion**: native C++'s reliable, architecture-independent advantage is in *not paying Python/framework process-startup cost* — not in out-computing a mature CPU runtime's matmul kernels. Warm-server throughput is very plausibly a losing or at-best-neutral battle against ONNX Runtime/PyTorch for this class of model; cold/single-shot invocation is a consistently large, easy-to-reproduce win.

**What this means for Sajal Labs going forward:**
- **Target use case narrows to**: serverless functions, CLI tools, edge/embedded devices invoked intermittently, batch/one-shot jobs — anywhere a fresh process (or fresh container) is paying import/load cost on (or near) every invocation. Explicitly **not** a target: high-throughput persistent inference servers handling sustained request volume, where warm latency and batching dominate and ONNX Runtime/vLLM-class systems are already strong (per landscape.md, related-work.md).
- **RQ5's answer, now evidence-backed rather than hypothesized**: native provides a meaningful advantage specifically for cold/single-request invocation, essentially independent of model architecture or size (at least across the two architectures and size range tested) — not for sustained compute-bound serving.
- **Numerical equivalence and dependency-footprint work remains exactly as valuable** — those findings (research/papers.md's identified gap around fp32 tolerance policy for native-compiled vs. framework-reference comparison) don't depend on which latency regime turns out to matter.
- **Kernel-level micro-optimization (the exp4-exp6 line of work) is now lower priority.** Six experiments spent chasing warm-latency parity for transformer-shaped models produced one negative result and no net improvement; that effort is better spent broadening cold-invocation evidence (more model types/sizes, per exp7's suggested next experiment) and building toward the actual compiler, where the win is now known to come from generating a small, dependency-free, fast-to-load artifact — not from generating a kernel-competitive one.
