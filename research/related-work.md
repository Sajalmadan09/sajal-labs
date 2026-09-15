# Related Work — LLM/GPU Serving Systems, Compilers, and Math Libraries

Researched from primary sources (official repos/READMEs, official docs, arXiv papers by project authors), September 2026.

---

## Part A: LLM Serving / GPU Inference Systems

### llama.cpp (+ ggml + GGUF)

Problem: Enable LLM/VLM inference "with minimal setup and state-of-the-art performance on a wide range of hardware — locally and in the cloud" — run transformers without a heavy framework stack.
Approach: Pure C/C++ on top of the ggml tensor library. ggml builds a computation graph at runtime, dispatches each op to a backend-specific kernel (CPU SIMD, CUDA, Metal, etc.). Models distributed as a single GGUF file (weights + tokenizer + metadata), memory-mapped at load time.
Language: C/C++, "without any dependencies."
Model formats: GGUF (successor to GGML/GGMF/GGJT). Python conversion scripts turn HF/safetensors/PyTorch checkpoints into GGUF; runtime only reads GGUF.
CPU: x86 AVX/AVX2/AVX512/AMX, ARM NEON, Apple Accelerate, RISC-V RVV. GPU: NVIDIA/AMD/Intel/Vulkan/WebGPU + vendor backends. Edge/mobile: prebuilt iOS/macOS/visionOS/tvOS xcframework; Android via JNI; runs on Raspberry Pi.
Quantization: very broad — legacy Q4_0/Q5_0/Q8_0, k-quants (mixed-precision per-block), I-quants (codebook/lattice-based) down to ~1.5 bits/weight.
Graph/kernel approach: no AOT compiler — graph built at load/run time, each node dispatched to a hand-written backend-specific kernel.
Serving: `llama-cli` (CLI), `llama-server` (OpenAI-compatible HTTP + web UI), `libllama`/`libggml` embeddable library — all three.
Python dependency: No — only for offline conversion scripts.
API surface: C API (`llama.h`); many third-party bindings, none required.

Strength: Dependency-free, single self-contained model file, runs everywhere from servers to phones to Raspberry Pi; unmatched breadth of community quantized models.
Weakness: Batching/scheduling simpler than vLLM/SGLang/TensorRT-LLM; almost exclusively decoder-transformer LLM/VLM focused; not a general tensor compiler or training framework.
What Sajal Labs could learn: The single-self-contained-file (mmap-loadable) + dependency-free C/C++ runtime + per-backend kernel dispatch pattern is exactly the shape a "compile any small model into a native artifact" project should imitate structurally.
What Sajal Labs should NOT reinvent: A new LLM/VLM quantized file format or C/C++ LLM inference runtime — GGUF + llama.cpp already own this thoroughly.
Source: github.com/ggml-org/llama.cpp; github.com/ggml-org/ggml docs/gguf.md; github.com/ggml-org/llama.cpp discussions/5063, /2094

### TensorRT / TensorRT-LLM

Problem: Maximize inference throughput/latency on NVIDIA GPUs by compiling a trained model into a hardware-specific, fused-kernel execution plan. TensorRT-LLM adds LLM-specific runtime features (KV cache, continuous batching, parallelism, speculative decoding, disaggregated prefill/decode).
Approach: AOT graph compilation — ONNX/PyTorch model fed to TensorRT builder, which does layer fusion, precision calibration, kernel auto-tuning (benchmarking candidates at build time) to produce a serialized "engine" for one specific GPU + TensorRT version. TensorRT-LLM: PyTorch-based, high-level Python `LLM` API + C++ runtime orchestration.
Language: TensorRT core C++ with Python bindings; TensorRT-LLM primarily Python with C++ runtime components.
Model formats: Input ONNX/PyTorch; output a proprietary serialized engine/plan tied to one GPU arch + TensorRT version (not portable).
CPU: none — GPU-only. GPU: NVIDIA only (Hopper, Blackwell, Ada, L4, RTX, etc.). Edge/mobile: Jetson via base TensorRT; TensorRT-LLM itself targets datacenter GPUs.
Quantization: INT8, FP8, FP4/NVFP4, AWQ, GPTQ with calibration; pre-quantized models on HF.
Graph/kernel approach: true AOT compiler — serialized, hardware-specific engine with fused/auto-tuned kernels chosen at build time.
Serving: Triton backend, TensorRT-LLM's own Python `LLM` API/OpenAI-compatible server, NVIDIA Dynamo. Base TensorRT has a lean C++ runtime (`trtexec` builds engines without Python) usable with zero Python at inference.
Python dependency: TensorRT-LLM high-level API — yes. Base TensorRT — no (C++ runtime API + `trtexec`).
API surface: C++ and Python (base TensorRT); primarily Python for TensorRT-LLM.

Strength: Best-in-class raw throughput/latency on NVIDIA GPUs via true AOT kernel fusion and hardware-specific tuning; deep quantization/parallelism support.
Weakness: NVIDIA-only; engines locked to specific GPU arch/TensorRT version (rebuild per target); heavy build-time dependency footprint even though runtime itself can be lean; poor fit for CPU/heterogeneous/edge.
What Sajal Labs could learn: The AOT "compile this exact model+shape into a hardware-specific fused-kernel artifact, benchmark kernel choices at build time" philosophy is the right compilation strategy to borrow conceptually (not GPU/NVIDIA-locked).
What Sajal Labs should NOT reinvent: Don't try to out-tune NVIDIA GPU kernel fusion for ops TensorRT already covers.
Source: github.com/NVIDIA/TensorRT-LLM; docs.nvidia.com/tensorrt-llm; docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide; docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/runtime-api

### vLLM

Problem: High-throughput, memory-efficient LLM serving under many concurrent requests — KV-cache memory fragmentation/waste and low GPU utilization of naive serving.
Approach: PagedAttention — KV cache in non-contiguous fixed-size "pages" (OS-paging analog) + continuous batching. V1 engine layers torch.compile with "piecewise" CUDA graphs (splitting around attention ops that aren't CUDA-graph-safe).
Language: Python core with custom CUDA/HIP/Triton kernels (C++/CUDA), some Rust for scheduling utilities.
Model formats: HF checkpoints (safetensors/PyTorch); can load GGUF-quantized weights.
CPU: yes, one of many backends. GPU: very broad — NVIDIA, AMD, Intel, TPU, Gaudi, Spyre, Ascend, Rebellions NPU, Apple Silicon, MetaX (widest hardware matrix surveyed). Edge/mobile: none meaningful.
Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO.
Graph/kernel approach: torch.compile JIT + piecewise CUDA-graph capture; compiled artifacts cached to disk to avoid recompiling on warm start.
Serving: OpenAI-compatible HTTP API (+ Anthropic Messages API, gRPC), or embedded Python library for offline/batch inference.
Python dependency: Yes, fully.
API surface: Python library + REST.

Strength: Broadest hardware backend coverage of any serving engine surveyed; PagedAttention now de facto industry standard; mature, fast-moving production ecosystem.
Weakness: Heavy dependency stack (PyTorch+CUDA+Python), not embeddable in a minimal native binary, non-trivial cold-start compile cost, large footprint.
What Sajal Labs could learn: PagedAttention-style paged memory management for variable-length sequences, and caching compiled kernel artifacts to amortize JIT cost across restarts.
What Sajal Labs should NOT reinvent: Large-scale, multi-backend, high-concurrency Python LLM serving with paged KV-cache scheduling.
Source: github.com/vllm-project/vllm; arxiv.org/abs/2309.06180 (PagedAttention); docs.vllm.ai/en/latest/design/{paged_attention,torch_compile}; vllm.ai/blog/2025-01-27-v1-alpha-release, 2025-08-20-torch-compile

### SGLang

Problem: Efficient, low-latency serving of complex, structured, multi-call LLM programs (agentic workflows) where naive serving redundantly recomputes shared prefixes.
Approach: Frontend DSL for structured LLM programs + RadixAttention — a radix-tree-indexed KV cache that auto-detects/reuses cached computation across requests sharing a prefix (LRU-evicted) — plus continuous batching, paged attention, speculative decoding, prefill-decode disaggregation, compressed FSM for structured-output decoding.
Language: Python core, Rust components (scheduler/tokenizer), custom CUDA/Triton kernels.
Model formats: HF-format checkpoints; quantized weights; some GGUF interop (far less central than llama.cpp).
CPU: some (Intel Xeon documented), primarily GPU-targeted. GPU: NVIDIA (GB200/B300/H100/A100/RTX 5090), AMD (MI300/MI355), Huawei Ascend, some TPU integration. Edge/mobile: none.
Quantization: FP4/FP8/INT4, AWQ, GPTQ.
Graph/kernel approach: explicitly "learned the design and reused code from" vLLM; CUDA graphs + custom Triton/CUDA kernels (notably RadixAttention), not a distinct AOT compiler.
Serving: HTTP server with OpenAI-compatible API + frontend DSL + Python engine/library mode.
Python dependency: Yes (Python + PyTorch).
API surface: Python (DSL + engine) and REST.

Strength: RadixAttention's shared-prefix cache reuse highly effective for agentic/multi-turn/structured-output workloads; strong adoption (claims 400,000+ GPUs).
Weakness: Same heavy Python/CUDA/PyTorch footprint as vLLM; substantial feature overlap with vLLM; no CPU-only/edge story.
What Sajal Labs could learn: Radix-tree KV-cache reuse for shared-prefix workloads; co-designing serving runtime with the program/DSL structure above it.
What Sajal Labs should NOT reinvent: Agentic/structured-generation-optimized GPU-cluster LLM serving with prefix caching.
Source: github.com/sgl-project/sglang; arxiv.org/abs/2312.07104; lmsys.org/blog/2024-01-17-sglang

### LMDeploy / TurboMind

Problem: Efficient compression, deployment, serving of LLMs (InternLM ecosystem roots), claiming higher throughput than baseline serving stacks for supported models.
Approach: Two interchangeable engines — TurboMind (highly optimized C++/CUDA, persistent/continuous batching, blocked KV cache) for max speed on well-supported models, and a pure-Python PyTorch engine for broader compatibility.
Language: TurboMind core C++/CUDA with pybind11; pipeline/server layer Python.
Model formats: HF checkpoints converted to TurboMind's internal layout at deploy time; AWQ-exported quantized weights.
CPU: none meaningful. GPU: NVIDIA V100+ (incl. RTX 50-series), ROCm, Huawei Ascend, Cambricon/MACA. Edge/mobile: none documented.
Quantization: AWQ (weight-only 4-bit), online INT8/INT4 KV-cache quant, MXFP4, llm-compressor 4-bit.
Graph/kernel approach: hand-written CUDA kernels + C++ persistent-batch scheduling and blocked KV-cache — not a general AOT graph compiler.
Serving: OpenAI-compatible HTTP server, Python pipeline API, multi-node proxy server, CLI.
Python dependency: Yes for pipeline/server layer even though TurboMind core is C++/CUDA; no documented zero-Python deployment path.
API surface: Primarily Python; TurboMind's C++ internals not exposed as a standalone embeddable library (unlike llama.cpp).

Strength: Strong throughput claims vs. vLLM for supported models; effective KV-cache quantization; pragmatic dual-engine speed/coverage tradeoff.
Weakness: Narrower model coverage than vLLM/TensorRT-LLM; C++ core not packaged as embeddable public library; Python layer still mandatory; smaller community.
What Sajal Labs could learn: The dual-engine strategy (one fast/narrow/hand-optimized native engine + one flexible broad-coverage fallback) balances speed against model-support breadth.
What Sajal Labs should NOT reinvent: Online int8/int4 KV-cache quantization kernels for GPU LLM serving.
Source: github.com/InternLM/lmdeploy; github.com/InternLM/lmdeploy docs/en/inference/turbomind.md; lmdeploy.readthedocs.io

### Candle (Rust, Hugging Face)

Problem: Minimalist Rust ML framework for inference (and light training) removing Python/GIL from production ML, enabling small fast-starting "serverless" binaries.
Approach: PyTorch-like eager tensor API implemented natively in Rust, shipped as a library a Rust binary links against directly.
Language: Rust.
Model formats: safetensors, PyTorch (.pt/.bin), NPZ, GGML/GGUF quantized formats (reuses llama.cpp-style quant types).
CPU: optimized backend, optional Intel MKL (x86) or Apple Accelerate. GPU: CUDA (+NCCL multi-GPU), Metal, WASM for browser. Edge/mobile: WASM/browser is closest analog; no dedicated Android/iOS packaging, but small binary size eases embedding.
Quantization: GGML/GGUF-style quantized tensor types (interops with llama.cpp schemes rather than defining its own).
Graph/kernel approach: no JIT/AOT compiler — dynamic eager execution, dispatching each op to a backend-specific kernel.
Serving: library only — no built-in HTTP server; community projects (candle-vllm) add OpenAI-compatible serving.
Python dependency: No — explicit design goal is Python-free production deployment.
API surface: Rust library API (PyTorch-like); `candle-pyo3` binding for tooling only.

Strength: Lightweight, fast-starting Rust binaries; large built-in model zoo (LLaMA, Mistral, Whisper, Stable Diffusion, YOLO, SAM).
Weakness: Much smaller ecosystem/serving maturity than vLLM/llama.cpp; no built-in high-concurrency server; narrower GPU backend coverage than llama.cpp.
What Sajal Labs could learn: Using Rust to get a zero-Python, single-binary ML deployment story, interoperating with llama.cpp-compatible quant formats rather than inventing a new one.
What Sajal Labs should NOT reinvent: A general-purpose Rust tensor/autograd library with broad model-loading support.
Source: github.com/huggingface/candle

### Burn (Rust, Tracel AI)

Problem: Unify training and inference so the exact trained code is the exact deployed code, avoiding "brittle and lossy" ONNX export round-trips, while staying portable to embedded/no_std targets.
Approach: Trait-based `Backend` abstraction (`Module`/`Tensor`/`Backend`) with composable decorators (Autodiff, Fusion, Remote); GPU/CPU backends via CubeCL, Burn's own cross-platform GPU-compute layer doing kernel fusion/JIT generation.
Language: Rust.
Model formats: `burn-onnx` (code-generates native Rust from ONNX, limited op coverage) and `burn-store` (PyTorch/safetensors weights); otherwise native Burn representation.
CPU: CubeCL CPU backend, ndarray backend, x86/ARM. GPU: CUDA, ROCm, Metal, Vulkan, WebGPU (LibTorch backend deprecated as of v0.22.0). Edge/mobile: no_std embedded + WASM targets (MNIST/image-classification demos).
Quantization: not a documented first-class feature; no GGUF-style low-bit scheme.
Graph/kernel approach: CubeCL performs real kernel fusion/JIT GPU kernel generation across backends — an actual compilation step, though still a general framework, not per-model AOT specialization.
Serving: no dedicated HTTP layer; a "Remote" backend supports client/server distributed compute, not LLM-style serving.
Python dependency: No — stated goal is "a single self-contained binary with no Python runtime to ship."
API surface: Rust library only; no official C/C++/Python bindings.

Strength: Genuinely unifies train/inference code paths; strong multi-backend portability including embedded/no_std/WASM; real kernel-fusion compiler (CubeCL).
Weakness: Young project, breaking changes still occur; limited ONNX op coverage; no LLM-serving/quantization ecosystem comparable to llama.cpp/vLLM; smaller pretrained-model zoo than Candle.
What Sajal Labs could learn: The Backend-trait + kernel-fusion-compiler (CubeCL) design for a portable, no_std-capable Rust ML runtime; philosophy of unifying training and inference code.
What Sajal Labs should NOT reinvent: A cross-backend Rust GPU-kernel-fusion compute layer from scratch, if a Rust-embedded approach is ever chosen.
Source: github.com/tracel-ai/burn

### Cross-cutting observations (Part A)

- **GGUF/llama.cpp is the closest existing analog to "single self-contained file + minimal C/C++ runtime, no Python"** — but architecturally specialized for decoder-style transformer LLMs/VLMs (metadata schema, kernel set, quant formats all built around attention/KV-cache/tokenizer). Not a generic "compile any small ONNX/PyTorch graph into a tiny C artifact" pipeline.
- **All four LLM-serving engines (TensorRT-LLM, vLLM, SGLang, LMDeploy) target GPU-cluster, high-concurrency serving of large transformer decoders.** Their core innovations (PagedAttention, RadixAttention, AOT engine compilation, blocked KV-cache) are specific to autoregressive decoding with large KV caches — irrelevant to a small classifier, embedding model, or tabular MLP with no KV cache and a single forward pass.
- **Candle and Burn are Rust frameworks you embed, not per-model AOT compilers.** Both ship "the framework" linked into your binary; neither generates a small, model-specific, dependency-free artifact tailored to one trained model's exact graph.
- **TensorRT is the one project here doing true per-model AOT specialization** (hardware-specific fused-kernel engine), but NVIDIA-GPU-only, heavy Python/PyTorch/CUDA build toolchain, engines locked to one GPU arch/version — poor fit for CPU-only or truly minimal embedded targets.
- **The "small/medium non-LLM model, minimal C artifact" space is not empty even outside this list.** Directly adjacent tools: **ONNX Runtime** (minimal-build C/C++ API, no Python at inference, arbitrary graphs — but a generic operator-dispatch interpreter, not per-model-specialized, not tiny by default); **TFLite/LiteRT** (+ microcontroller variant); **m2cgen** (converts trained classical-ML models — linear models, tree ensembles — directly into plain C/Java/Go source, the closest existing thing to genuine per-model AOT C codegen for small models, but limited to classical ML, not neural nets/transformers).
- **The most defensible gap, stated conservatively:** true per-model AOT specialization (TensorRT's philosophy) producing a tiny, dependency-free, CPU-only C artifact (not GPU-locked, not a general interpreter you still link) for small/medium non-LLM neural models. ONNX Runtime-minimal is general-but-not-specialized; m2cgen is specialized-but-classical-ML-only; llama.cpp/ggml is specialized-but-LLM-shaped. The intersection — neural-net-capable + per-model AOT-specialized + truly minimal C — is narrower and more defensible than "no one solves small-model inference" (which is false).
- **Python dependency at inference is the clearest differentiator across this set:** llama.cpp, Candle, Burn, and base TensorRT have none; vLLM, SGLang, and (practically) LMDeploy/TurboMind all require Python/PyTorch at inference. Any "no Python at inference" pitch must be benchmarked against llama.cpp/Candle/Burn/ONNX Runtime/TFLite — those already concede the point — not against the four Python-bound LLM servers.

---

## Part B: ML Compilers & Math/Kernel Libraries

### Apache TVM

Problem it solves: Compiling trained ML models into optimized, deployable code across many diverse hardware backends without hand-writing kernels per target.
Approach: Multi-level IR compiler stack. High-level graph IR (Relax) represents model structure with control flow; progressively lowers to TensorIR (explicit loop nests, memory, threading), then to target-specific code (LLVM, CUDA, etc.). Auto-tuning/auto-scheduling (AutoTVM/Ansor/MetaSchedule) searches for fast schedules per target.
Language: Core C++; primary user interface Python.
Operators: full NN operator set as tensor programs (conv, matmul, elementwise, reductions, control flow via Relax); can call vendor libraries (cuBLAS, CUTLASS, cuDNN) for select ops.
Hardware: x86, ARM CPUs, NVIDIA GPUs (CUDA), OpenCL devices, pluggable target system; microcontroller/embedded via microTVM.
Usage model: compiler framework invoked from Python — build IRModule, apply passes, produce a `runtime.Module` loaded/executed by TVM's runtime. AOT/JIT pipeline, not a linked math library.

Strength: Genuinely full-stack (import, graph+loop optimization, auto-tuning, multi-backend codegen); mature, widely deployed in production.
Weakness: Heavy dependency (LLVM, Python tooling), steep learning curve, auto-tuning can be slow; less turnkey than calling a BLAS library for a handful of fast ops.
What Sajal Labs could learn: The layered-IR idea (graph IR → loop IR → target code) — separate "what to compute" from "how it's scheduled" so schedule choices are swappable without rewriting operator semantics.
What Sajal Labs should NOT reinvent: A general auto-scheduler/auto-tuning search engine, or a retargetable multi-backend codegen framework.
Source: tvm.apache.org/docs/{arch/index,}; github.com/apache/tvm

### MLIR

Problem it solves: Compiler-construction fragmentation — every domain built its own one-off IR; MLIR gives a common, reusable, extensible IR framework so infrastructure can be shared and heterogeneous hardware targeted more cheaply.
Approach: "Dialect"-based architecture — extensible composable op/type sets (`affine`, `linalg`, `gpu`, `llvm`) coexisting in one IR, progressively lowered/converted into each other. Reusable infra: pass management, pattern rewriting, verification, SSA representation with regions.
Language: C++ (LLVM sub-project); Python and C bindings.
Operators: not itself an operator set — a meta-framework. Concrete dialects (`linalg`, `tosa`, `stablehlo` used downstream) define ML ops; MLIR ships no fixed ML op library.
Hardware: whatever a dialect targets (GPUs via NVVM/ROCm, CPUs via LLVM dialect, custom accelerators via user dialects). Stops before final machine-code gen (register allocation, instruction scheduling) — that's still LLVM's job.
Usage model: compiler infrastructure/library — embed MLIR, define/reuse dialects, write lowering passes, emit LLVM IR or another backend IR. Infrastructure, not an end-user compiler or linkable math library.

Strength: Extremely reusable/composable; de facto substrate for a huge swath of the ML-compiler ecosystem (TensorFlow, IREE, Triton, torch-mlir).
Weakness: Infrastructure, not a solution — still must design dialects and write all lowering passes; no built-in "compile my model" story without adopting something like IREE/TVM/torch-mlir on top.
What Sajal Labs could learn: The dialect/progressive-lowering pattern — representing a model at several abstraction levels and lowering step-by-step — is useful even without adopting MLIR itself; the discipline of strict IR verification between passes.
What Sajal Labs should NOT reinvent: A general SSA-based, dialect-extensible compiler IR framework with pass infrastructure — if a "real" IR is ever needed, adopt MLIR/IREE rather than growing one organically.
Source: mlir.llvm.org; mlir.llvm.org/docs

### IREE

Problem it solves: Deploying ML models efficiently across an extremely broad hardware range — datacenter GPUs down to mobile/embedded/bare-metal — from a single compiled artifact, minimal runtime footprint.
Approach: MLIR-based AOT whole-program compiler. Embeds both scheduling logic (coordinating work across parallel hardware) and execution logic (target-specific compiled binaries/kernels) into one compiled module, rather than keeping a big interpreter/graph-executor at runtime.
Language: Compiler C++ (MLIR-based); runtime C (embeddable/portable); Python bindings for driving compilation.
Operators: whatever the input model expresses (imported from TF, TFLite, JAX, PyTorch, ONNX via MLIR import paths).
Hardware: CPUs (x86, ARM, RISC-V via LLVM codegen), GPUs via Vulkan/SPIR-V, CUDA, ROCm/HIP, Metal (Apple GPU), WebGPU, accelerator backends (AMD AIE); platforms: Linux, Windows, macOS, Android, iOS, bare metal, WASM.
Usage model: two-phase — compile AOT on a host machine into a deployable module, embed IREE's small runtime (reported as low as ~30KB) via its C API to load/execute. "Compiler + minimal embeddable runtime," not a plain linked math library or full training-framework runtime.

Strength: Genuinely designed for the small/embedded end (unlike TVM, which skews server/cloud/research), real multi-backend GPU support including Apple Metal; single AOT-compiled artifact simplifies deployment.
Weakness: Inherits MLIR's complexity/build weight (LLVM toolchain, MLIR dialects); heavyweight dependency for a from-scratch minimal-C/C++ generator; less mature/ubiquitous than TVM or vendor libraries.
What Sajal Labs could learn: The "compile once, embed a tiny runtime" deployment model is exactly Sajal Labs' target shape — worth studying IREE's module format/VM bytecode design even without depending on it.
What Sajal Labs should NOT reinvent: A portable GPU/accelerator backend abstraction (Vulkan/Metal/CUDA/ROCm from one IR) — lean on IREE/MLIR rather than rebuild if GPU portability is ever needed.
Source: iree.dev; github.com/iree-org/iree; iree.dev/guides/ml-frameworks

### OpenBLAS

Problem it solves: Free, high-performance BLAS/LAPACK implementation, competitive with vendor-tuned libraries, without licensing restrictions.
Approach: Optimized BLAS based on GotoBLAS2; performance from hand-written CPU-microarchitecture-specific kernels (often assembly) plus runtime CPU-feature detection (`DYNAMIC_ARCH=1`) to pick the best kernel for the detected chip.
Language: C and Fortran with hand-optimized assembly kernels per architecture.
Operators: full BLAS Levels 1–3 (incl. GEMM) plus LAPACK on top.
Hardware: x86/x86-64 (many microarchitectures), ARM/ARM64, MIPS, PowerPC, IBM zSeries, RISC-V, LoongArch64, WASM. **No first-class Apple-Silicon-specific tuned path comparable to Accelerate's AMX use** — runs on Apple Silicon via generic ARM64/NEON, not Apple's proprietary matrix units. No NVIDIA GPU support — CPU-only.
Usage model: classic linked library — link statically/dynamically, call standard BLAS/LAPACK signatures from C/C++/Fortran. No compiler step; it's the callee.

Strength: Mature, portable, free (BSD), near-vendor performance on many CPU architectures, drop-in BLAS-API compatibility.
Weakness: Architecture-specific kernel tuning maintained by community — new CPU generations lag before hand-tuned kernels appear; on Apple Silicon specifically does not exploit the AMX coprocessor the way Accelerate does, so typically slower than Accelerate on Mac.
What Sajal Labs could learn: The "generic fallback + runtime dispatch to hand-tuned per-CPU kernel" pattern (`DYNAMIC_ARCH`) is a good template for structuring hot loops (naive scalar fallback, then feature-detected SIMD kernel).
What Sajal Labs should NOT reinvent: Hand-tuned Level-3 BLAS (GEMM) kernels for x86/ARM — link OpenBLAS (or platform equivalent) rather than hand-rolling.
Source: github.com/OpenMathLib/OpenBLAS (README.md)

### oneDNN (Intel)

Problem it solves: Gives DL frameworks/inference engines optimized, portable NN "building block" primitives so they don't hand-write/hand-tune conv/matmul/normalization kernels per CPU/GPU generation.
Approach: Primitive-based API (create a primitive descriptor for an op + memory format, let the library JIT-pick an optimized implementation for detected hardware, execute). Implements the oneAPI spec; internally JIT-compiles kernels tuned to the detected ISA (AVX-512, AMX on Intel; SVE on Arm).
Language: C++ (C++11) with a C API.
Operators: convolution (fwd/bwd), inner product, matmul, RNN (LSTM/vanilla RNN/GRU), batch/layer/group/local-response normalization, pooling, resampling, eltwise/activations, binary ops, softmax, PReLU, reduction, sum, concat, shuffle, reorder.
Hardware: primary Intel 64/AMD64 and Arm AArch64 CPUs (tuned for Xeon Scalable, Core Ultra, Arm Neoverse); GPU: Intel Arc/Data Center primary, experimental NVIDIA/AMD; experimental POWER, z/Architecture, RISC-V.
Usage model: linked C++ library with primitive-construction API — frameworks (PyTorch, TF, ONNX Runtime, OpenVINO) integrate it internally; end users rarely call it directly.

Strength: Covers full DL-primitive surface (not just BLAS) — normalization, RNN, pooling, elementwise fusions — with runtime JIT tuning; standard CPU backend for major frameworks.
Weakness: Tuning heavily Intel-hardware-centric; Arm/GPU/other-vendor support present but secondary; adds real integration weight (primitive descriptors, memory format management) beyond a plain BLAS call.
What Sajal Labs could learn: A primitive catalog broader than BLAS — conv/norm/pooling/RNN as first-class fused-capable ops — is closer to what a model-specific codegen tool needs than raw BLAS; the memory-format abstraction (blocked/packed layouts per hardware) is worth studying even for a smaller subset.
What Sajal Labs should NOT reinvent: JIT-tuned, ISA-specific conv/GEMM/normalization kernels for x86 (AVX2/AVX-512/AMX).
Source: github.com/uxlfoundation/oneDNN; uxlfoundation.github.io/oneDNN/supported_primitives.html

### Eigen

Problem it solves: Clean, expressive, high-performance C++ linear algebra without a separate build/link step or heavyweight dependency.
Approach: Pure header-only template library using expression templates — operations build compile-time expression trees, compiler generates fused, specialized code rather than materializing temporaries. Implements its own explicit SIMD vectorization layer rather than relying on the compiler's auto-vectorizer.
Language: C++ (template metaprogramming), no dependency beyond the standard library.
Operators: dense/sparse matrix-vector arithmetic, decompositions/solvers (LU, QR, Cholesky, SVD, eigenvalues), geometric transforms — general linear algebra, no DL-specific primitives (no conv/normalization).
Hardware: CPU only — SSE2/3/4, AVX, AVX2, FMA, AVX-512 (x86), ARM NEON (incl. Apple Silicon), PowerPC AltiVec/VSX, IBM ZVector, MIPS MSA, with scalar fallback. No GPU.
Usage model: header-only include — no binary to build/link, just `#include`. Simplest integration model surveyed.

Strength: Zero build/link friction, portable across CPU ISAs incl. Apple Silicon NEON, ergonomic C++ API, mature/trusted (ROS, TensorFlow, Ceres).
Weakness: Not a DL-primitive library (no conv/pooling/normalization, no autodiff, no GPU); hand-rolled kernels generally don't beat vendor-tuned BLAS at scale for large GEMMs, though fine for small/medium fixed-size matrices.
What Sajal Labs could learn: The "header-only, zero-link-step" integration model is directly relevant — a generated minimal C/C++ artifact wants exactly this kind of dependency (or none at all).
What Sajal Labs should NOT reinvent: General dense linear-algebra plumbing (matrix expression handling, decompositions/solvers) if generated code ever needs more than raw GEMM/conv.
Source: eigen.tuxfamily.org / libeigen.gitlab.io; libeigen.gitlab.io/pages/faq

### Apple Accelerate framework

Problem it solves: Apple-hardware-accelerated math (linear algebra, DSP, image processing, NN primitives) through one system framework, tuned for Apple Silicon's CPU vector units, AMX matrix coprocessor, GPU/Neural Engine.
Approach: Umbrella framework of sub-libraries: BLAS and LAPACK, vDSP (vectorized signal processing/array arithmetic, FFTs), vImage (image processing), BNNS (Basic Neural Network Subroutines — layers, activations, loss functions for small NNs). Internal dispatch to the best execution unit (vector CPU, AMX, or GPU) is not publicly documented in detail.
Language: C API core, with Objective-C and Swift overlays.
Operators: full BLAS (levels 1–3) and LAPACK; vDSP vector arithmetic/FFTs/filtering; vImage pixel ops; BNNS NN layer types, activations, loss functions, inference building blocks.
Hardware: Apple Silicon (and legacy Intel Mac) CPUs, AMX acceleration for BLAS/BNNS-style workloads, GPU dispatch for some ops; macOS, iOS, iPadOS, watchOS, tvOS, visionOS.
Usage model: linked system framework — already present on every Apple OS install, no separate distribution/versioning burden.

Strength: Free, zero-install, the only public API to Apple's AMX matrix-multiply hardware — on Apple Silicon likely the fastest available CPU BLAS without custom Metal kernels; covers linear algebra + basic NN primitives (BNNS) + DSP in one place. (Caveat: this is Apple's own positioning — we have Apple Silicon hardware to test on, unlike the NVIDIA stack, but haven't yet benchmarked it ourselves.)
Weakness: Apple-platform-only (no Linux/Windows/Android portability); undocumented internal dispatch heuristics (AMX vs. vector CPU vs. GPU) — something of a black box; BNNS fairly basic vs. a full DL-inference engine (no broad model-format/graph support).
What Sajal Labs could learn: The "one umbrella framework, always present, dispatches to the right execution unit automatically" model is the ideal UX to mirror at the API-design level.
What Sajal Labs should NOT reinvent: Any code competing with Apple's AMX coprocessor use — no public/documented instruction interface exists; use Accelerate as the BLAS backend on this hardware.
Source: developer.apple.com/documentation/accelerate (+ BLAS/vDSP/BNNS subpages)

### NVIDIA CUDA / cuBLAS / CUTLASS

Problem it solves: CUDA — NVIDIA's general GPU parallel-programming platform. cuBLAS — ready-to-call vendor-optimized BLAS on CUDA. CUTLASS — CUDA C++ template building blocks for developers writing custom high-performance GEMM/conv kernels cuBLAS's fixed API can't express.
Approach: cuBLAS — host-callable library API (`cublas_v2.h`): allocate GPU memory, copy data, call functions dispatching to pre-tuned internal kernels, copy results back (plus cuBLASLt for fusable GEMM, cuBLASXt for multi-GPU). CUTLASS — header-only CUDA C++17 template library decomposing GEMM into a hierarchy (thread → warp → threadblock → device-wide), letting developers instantiate custom kernels at compile time (tile sizes/dtypes/precision); newer versions add a CuTe Python DSL.
Language: CUDA C/C++ (cuBLAS compiled library; CUTLASS C++17 templates, plus a Python DSL).
Operators: cuBLAS — full BLAS Levels 1–3, batched/mixed/low-precision variants. CUTLASS — GEMM and GEMM-like/convolution across FP64/FP32/FP16/BF16/FP8/FP4/integer, tensor-core-optimized.
Hardware: NVIDIA GPUs only. cuBLAS: compute capability 5.0+, Tensor Cores from Volta on. CUTLASS: Volta (SM70) through Blackwell (SM100/SM120).
Usage model: cuBLAS = classic linked library (closest analog to OpenBLAS/Accelerate, but GPU-resident). CUTLASS = header-only templates compiled into your own CUDA source — closer to Eigen's model, used for custom/fused kernels beyond cuBLAS's fixed API.

Strength (per NVIDIA's own docs/claims): cuBLAS — turnkey, heavily tuned, near-optimal throughput utilization, standard drop-in API. CUTLASS — exposes the full tensor-core/mixed-precision surface for custom fused kernels while claiming near-optimal throughput.
Weakness: NVIDIA-GPU-only, CUDA-toolchain-only, zero portability to Apple Silicon or non-NVIDIA hardware; CUTLASS has a steep learning curve requiring real GPU kernel expertise.
What Sajal Labs could learn: The cuBLAS/CUTLASS split — simple linked-library API for most users, template toolkit for the few needing custom fused kernels — is a good API-tiering model to imitate for any backend, even CPU-only.
What Sajal Labs should NOT reinvent: GPU tensor-core kernel programming for NVIDIA hardware — out of scope on our Apple Silicon dev hardware regardless, and a deeply specialized, ecosystem-scale effort even with GPU access.

**Important caveat: our team's hardware is Apple Silicon with no NVIDIA GPU.** Everything above about cuBLAS/CUTLASS is taken from NVIDIA's own documentation/claims — unverified and unbenchmarked by us. Treat as background/landscape knowledge only, not a near-term design target.
Source: docs.nvidia.com/cuda/cublas; github.com/NVIDIA/cutlass (README.md)

### Cross-cutting observations (Part B)

- Two clearly different "levels" of tool exist: **(1) linked math libraries with a fixed, narrow API** (OpenBLAS, Accelerate, cuBLAS, oneDNN's primitive API — call a function, they run a kernel) vs. **(2) full compiler stacks** (TVM, MLIR, IREE — feed a whole model graph, get new code/artifacts back). Eigen and CUTLASS sit in between: header-only, compile-time-specialized template libraries needing a C++ compiler pass but no separate IR/graph-level optimization.
- For a project generating **model-specific native C/C++ inference code** (Sajal Labs' actual goal), **option (1) — calling an existing BLAS-like library for heavy matmul/conv — is the far better starting point** than adopting a full compiler stack:
  - The hard, narrow, high-effort part (competitive GEMM/conv kernels per CPU ISA) is exactly what OpenBLAS/oneDNN/Accelerate/Eigen already solve; Sajal Labs' value-add is codegen, not kernel engineering.
  - An artifact that just `#include`s a header (Eigen) or links a small library (Accelerate/OpenBLAS) stays minimal and dependency-light; depending on MLIR/IREE/TVM as a build dependency works against that (LLVM-scale toolchains, large binaries).
  - A full compiler stack (2) only pays off once multi-backend targeting from one IR, automatic fusion/scheduling search, or ingesting arbitrary graphs from many frameworks is actually needed — none of which is required to hand-roll per-model C/C++ calling into a math library.
  - A pragmatic middle path: borrow TVM/IREE's "graph IR → tiled loop IR → target code" *idea* without adopting them as dependencies.
- **On Apple Silicon (our actual dev hardware), the most practical native math backend to start with is Apple's Accelerate framework**: ships with the OS (zero install/versioning burden), free, the only public path to the AMX matrix coprocessor — very likely faster than generic OpenBLAS on the same machine for BLAS-shaped workloads (per Apple's positioning, not yet independently benchmarked by us). Its BNNS component also covers basic NN primitives directly, closer to what an inference artifact needs than raw BLAS.
- **Eigen is the natural complement to Accelerate**: header-only (no link step), already NEON-vectorized for Apple Silicon, useful for linear algebra outside Accelerate's fixed API or for portability to non-Apple targets.
- **OpenBLAS is the fallback for portability off Apple hardware** (Linux/Windows/other-ARM) but unlikely to beat Accelerate specifically on Apple Silicon, since it has no AMX access.
- **oneDNN is Intel-hardware-centric in tuning depth** — on Apple Silicon it would run generic AArch64 paths without Intel-specific JIT advantages; lower priority than Accelerate for this team's hardware, worth revisiting for x86 targets later.
- **Recommended starting posture**: generate C/C++ that calls Accelerate (BLAS/BNNS) on macOS/iOS, with Eigen as a header-only fallback/complement, and defer any full compiler-stack (MLIR/IREE/TVM) adoption until there's a concrete need for multi-backend targeting or automatic kernel search that hand-written codegen can't satisfy.
