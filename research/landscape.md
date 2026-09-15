# Native/Edge Inference Runtime Landscape

Research conducted via official docs, official GitHub repos, and official engineering blogs only (Sept 2026). Gaps genuinely not confirmed in primary sources are marked "unknown" rather than guessed.

## ONNX Runtime

Problem:
Provide a single cross-platform, cross-framework inference accelerator so a model trained in any framework (PyTorch, TF/Keras, scikit-learn, etc.) that can export to ONNX can run performantly on any hardware without retraining or framework lock-in.

Approach:
Load an ONNX graph, apply hardware-agnostic graph optimizations, then partition the graph into subgraphs assigned to "Execution Providers" (EPs) — pluggable hardware-specific backends (CUDA, TensorRT, OpenVINO EP, oneDNN, ROCm, QNN, CoreML, NNAPI, WebGPU/WebNN, etc.). Falls back to a default CPU EP for unsupported ops.

Language:
Core is C++. Official bindings: C, C++, Python, C#, Java, JavaScript (Node/Web/React Native), Objective-C.

Model formats supported:
ONNX (.onnx) natively; also its own optimized "ORT format" (serialized, pre-optimized) for mobile/web. Anything convertible to ONNX (PyTorch, TF, TFLite, scikit-learn via onnx converters) is supported indirectly.

CPU support: Yes, default EP, x86/x64/ARM, with oneDNN/MLAS kernels.
GPU support: Yes — CUDA, TensorRT, ROCm (AMD), DirectML (Windows), CoreML (Apple GPU), WebGPU (browser).
Edge/mobile support: Yes — dedicated mobile package, Android (NNAPI, QNN, XNNPACK EP) and iOS (CoreML) EPs, reduced/custom builds for size-constrained targets.
Quantization support: Yes — dynamic, static, and QAT-produced quantized models via QDQ format; EPs (e.g. TensorRT) fuse QDQ nodes for hardware int8 paths.
Graph compilation: Graph-level optimizations (constant folding, node fusion, layout transforms) plus hardware partitioning into EP subgraphs; no full AOT machine-code compilation of the whole model (delegated to EPs like TensorRT).
Serving story: Primarily a library/SDK; no official first-party HTTP server (Triton has an ORT backend, but that's NVIDIA's project). ONNX Runtime GenAI adds a higher-level generate() API for LLMs.
Python dependency: No — Python only used for model prep/quantization tooling.
API surface: C API is the stable ABI; C++, Python, C#, Java, JS, Objective-C on top.

Strength: Extremely broad hardware/EP ecosystem and framework-agnostic ONNX standard; mature reduced-build tooling for shrinking binaries to just the operators a model needs.
Weakness: General-purpose core (must support any ONNX graph); truly minimal model-specific binaries require manual custom-build steps rather than being the default.
What Sajal Labs could learn: The EP abstraction and the reduced-operator-config-file mechanism (statically strip to only kernels a model config needs) are proven prior art for "minimal model-specific artifact."
What Sajal Labs should NOT reinvent: The ONNX format itself and its multi-framework/multi-hardware EP breadth.
Source: onnxruntime.ai/docs/{performance/model-optimizations/quantization, execution-providers/QNN-ExecutionProvider, build/custom, build/inferencing, reference/operators/reduced-operator-config-file}; github.com/microsoft/onnxruntime docs/Reduced_Operator_Kernel_build.md

## OpenVINO (Intel)

Problem: Optimize/deploy trained models efficiently on Intel hardware (CPU, iGPU/dGPU, NPU) for edge/server inference behind one API.
Approach: Convert to an Intermediate Representation (IR); plugin architecture routes inference to device-specific plugins (CPU/GPU/NPU) implementing a common runtime API.
Language: C++ core, C and Python bindings, Node.js/JS package.
Model formats: PyTorch, TensorFlow, TFLite, ONNX, PaddlePaddle, JAX/Flax — run directly or converted to OpenVINO IR (.xml/.bin).
CPU: Intel x86 and Arm CPUs. GPU: Intel integrated/discrete (oneAPI/Level Zero). Edge: industrial/PC edge (embedded Linux/Yocto, Docker), NPU — not phone edge like TFLite/ExecuTorch.
Quantization: Yes — NNCF: PTQ, QAT, weight compression (incl. 4-bit weight-only for LLMs), sparsity.
Graph compilation: Conversion-time graph optimization/fusion; device plugins further compile IR at load time.
Serving: Both a library AND OpenVINO Model Server (OVMS) — a standalone C++ server with TensorFlow-Serving-compatible REST/gRPC, Kubernetes-ready. One of the few projects surveyed with an actively maintained first-party server.
Python dependency: No — C++ API fully capable standalone; Python is most feature-complete dev API.
API surface: C++, Python, C, JS/Node.

Strength: Actively maintained first-party production server (OVMS); deep Intel hardware optimization; mature NNCF quantization.
Weakness: Intel-hardware-centric; IR + plugin + OVMS layering adds real operational surface vs. "link a static lib."
What Sajal Labs could learn: Clean separation of offline conversion/optimization tool vs. device-specific runtime plugin; validates that a first-party server binary alongside the library is worth shipping.
What Sajal Labs should NOT reinvent: x86/Intel low-level kernel optimization (oneDNN-equivalent), or a TF-Serving-compatible REST/gRPC protocol from scratch.
Source: docs.openvino.ai/2024/{about-openvino, openvino-workflow/running-inference}; docs.openvino.ai/2025/model-server/{ovms_what_is_openvino_model_server, ovms_docs_serving_model}; github.com/openvinotoolkit/model_server

## ExecuTorch (PyTorch Edge)

Problem: Deploy PyTorch models directly to edge/mobile/embedded devices with a lightweight C++ runtime and minimal binary footprint, replacing deprecated PyTorch Mobile/TorchScript-on-device.
Approach: AOT pipeline on the PyTorch 2 compiler stack: torch.export() → ATen Dialect → Core ATen Dialect → Edge Dialect (dtype/layout info) → Backend Dialect (hardware delegate partitioning) → serialized FlatBuffer `.pte`. A small separate C++ runtime loads/executes the `.pte` with only linked-in kernels/backends.
Language: C++ runtime core; Python only for export/AOT (via PyTorch).
Model formats: PyTorch via torch.export() → `.pte`. Not a general multi-framework format.
CPU: portable/optimized kernels + XNNPACK delegate. GPU: via delegates (Core ML/Apple GPU-ANE, partial Vulkan) — not first-class universal GPU. Edge/mobile: primary purpose — iOS, Android, embedded/bare-metal MCUs; delegates for Core ML, QNN, Arm TOSA/Ethos, MediaTek, XNNPACK.
Quantization: Yes — PTQ and QAT via PyTorch 2 quantization APIs, targeting edge size reduction.
Graph compilation: Genuine AOT lowering pipeline through the dialects above, with backend-specific partitioning/delegation and user compiler passes for fusion/memory planning.
Serving: Library + CLI example runner (executor_runner); no HTTP server — explicitly an on-device embedded runtime, not a serving product.
Python dependency: No at inference — only for export.
API surface: C++ runtime; Python export tooling; ecosystem Swift/Kotlin wrappers.

Strength: Genuinely minimal core — documented ~50KB base runtime (no operators linked), selective build links in only the operators/kernels a given `.pte` uses. Closest existing prior art to "model-specific minimal runtime."
Weakness: Locked to PyTorch as source framework (via torch.export); young project, evolving dialect/delegate stack vs. TFLite/ONNX Runtime's maturity.
What Sajal Labs could learn: The AOT dialect-lowering pipeline (general → device-specific → backend-specific) and especially the selective-build mechanism (link only what a compiled program needs) is the closest prior art to Sajal Labs' stated core idea. Note: `huggingface/optimum-executorch` already does model+export-to-single-`.pte` distribution from the Hub for select architectures (Llama, Gemma, Qwen, OLMo).
What Sajal Labs should NOT reinvent: torch.export/AOTAutograd compiler machinery and PyTorch's quantization APIs.
Source: docs.pytorch.org/executorch/stable/{getting-started-architecture, intro-how-it-works}; github.com/pytorch/executorch/discussions/13678; github.com/huggingface/optimum-executorch; huggingface.co/docs/optimum/en/exporters/executorch/overview

## XNNPACK

Problem: Provide highly-optimized, portable, low-level NN operator kernels so higher-level frameworks don't each hand-write SIMD kernels per CPU architecture.
Approach: A pure operator library (not a standalone inference engine, no own model format). Frameworks call its operator/Subgraph APIs from their own graph representations.
Language: C (C11) core, C++17 in parts of build; C API is the integration surface.
Model formats: None directly — has a Subgraph API for small ad-hoc graphs, doesn't parse ONNX/TFLite files itself.
CPU: ARM64, ARMv7 (NEON), ARMv6, x86/x86-64 (up to AVX512), WASM (MVP/SIMD/Relaxed SIMD), RISC-V, Hexagon (HVX). GPU: none — CPU/SIMD/DSP-only. Edge/mobile: primary design target (mobile, server, Web via WASM).
Quantization: Yes — INT8 kernels alongside FP32/FP16.
Graph compilation: Minimal — its Subgraph API does light fusion/execution for small graphs passed to it; no general model-graph compiler.
Serving: Pure library, no CLI, no server.
Python dependency: No — build-time Python 3 only for build scripts.
API surface: C API.

Strength: Extremely broad, battle-tested SIMD kernel coverage across nearly every CPU/DSP architecture in production — it's the CPU backend inside TFLite, PyTorch Mobile/ExecuTorch, ONNX Runtime, TensorFlow.js, MediaPipe.
Weakness: Deliberately no model-format parsing, no graph IR beyond small ad-hoc subgraphs, no GPU support — solves exactly one layer (fast operators).
What Sajal Labs could learn: The layering discipline — a hyper-focused kernel library many different runtimes embed rather than reinventing SIMD kernels — validates "kernel library" and "runtime/format layer" as separate concerns.
What Sajal Labs should NOT reinvent: Hand-rolled SIMD CPU kernels for conv/matmul/pooling across ARM/x86/WASM/RISC-V.
Source: github.com/google/XNNPACK README.md

## MNN (Alibaba)

Problem: Give Alibaba apps (Taobao, Tmall, Youku) and general devs a fast, lightweight on-device inference (and lightweight training) engine, battle-tested at scale, including on-device LLM/diffusion workloads.
Approach: MNN-Converter ingests other frameworks' models, applies graph optimization, emits MNN's own format; MNN runtime executes via heavily hand-optimized assembly kernels. Satellite tools: MNN-Compress (quantization/pruning), MNN-Express (control flow), MNN-CV, MNN-Train, MNN-LLM.
Language: C++ core; official Python and Java (Android) bindings.
Model formats: Converts from TensorFlow, TFLite, Caffe, ONNX, TorchScript (178 TF ops, 52 Caffe ops, 163 TorchScript ops, 158 ONNX ops per repo) into MNN's native format.
CPU: ARM (v7a, v8), x86/x64 (SSE4.1, AVX2, AVX512), hand-written assembly. GPU: Metal (iOS), OpenCL, Vulkan, CUDA. Edge/mobile: core focus — iOS 8+, Android 4.3+, POSIX-embedded; NPU backends (CoreML, HiAI, NNAPI, QNN).
Quantization: FP16, BF16, INT8 — documented 50–70% model-size reduction.
Graph compilation: MNN-Converter does graph optimization/fusion at conversion time.
Serving: Library (+ Python API); MNN-LLM adds on-device LLM runtime (Qwen/Baichuan/LLaMA) + diffusion support. No official HTTP server.
Python dependency: No — compiled models run standalone (C++/Java); Python is convenience/tooling only.
API surface: C++ core, Python, Java (Android).

Strength: Concrete, published minimal-binary numbers: ~800KB Android core (armv7a), ~12MB full iOS static lib, and `MNN_BUILD_MINI` mode trading dynamic shapes for ~25% further size reduction.
Weakness: Proprietary format + converter with finite (if broad) op coverage; less standardized than ONNX.
What Sajal Labs could learn: The published binary-size numbers are a good benchmark bar; "fixed-shape minimal build" trade-off (give up dynamic shapes for a smaller, model-specific binary) is directly relevant prior art.
What Sajal Labs should NOT reinvent: Hand-tuned ARM/x86 assembly kernels and a mature multi-framework converter.
Source: github.com/alibaba/MNN

## ncnn (Tencent)

Problem: Run NNs with high performance directly on mobile/embedded/desktop inside Tencent apps (QQ, WeChat, Qzone, Pitu) with zero third-party runtime dependencies — maximum portability, minimal deployment friction.
Approach: Standalone C++ engine with its own compact format (.param + .bin), populated via pnnx (recommended PyTorch/ONNX converter) or legacy Caffe/MXNet/Darknet converters. No BLAS/NNPACK dependency — self-contained math, explicit blob/workspace memory allocator design for low, predictable footprint.
Language: Pure C++ core, C API wrapper, full Python bindings.
Model formats: Native .param/.bin, reached via pnnx (PyTorch, ONNX) or legacy converters. Netron-compatible.
CPU: strong ARM NEON + multi-core scheduling; also x86, MIPS, RISC-V, LoongArch. GPU: Vulkan (cross-vendor, not vendor-locked). Edge/mobile: core focus — official packages for Android, iOS, HarmonyOS, macOS, Linux, Windows, WASM, Raspberry Pi, Jetson.
Quantization: INT8 + FP16 storage/arithmetic, `ncnnoptimize` tool for post-conversion optimization.
Graph compilation: Conversion-time optimization (converters + ncnnoptimize), not a JIT/AOT compiler; multi-input/output/branch graphs supported natively.
Serving: Pure library + CLI tools (ncnnoptimize, onnx2ncnn). No official server.
Python dependency: No — zero third-party runtime deps including Python by design; Python binding is tooling convenience only.
API surface: C++ primary, C API, Python, extensible custom-layer plugin system.

Strength: Among the most dependency-free runtimes surveyed (no BLAS/NNPACK/third-party deps, pure C++) plus vendor-neutral Vulkan GPU support (works across Android GPU vendors).
Weakness: Proprietary format needing conversion, with the usual op-coverage gaps; smaller ecosystem than ONNX Runtime/TFLite.
What Sajal Labs could learn: "Zero third-party runtime dependency, pure C/C++" is achievable and already shipped at massive scale — dependency-light is a provable design point, not marketing. Blob/workspace allocator design is a good memory reference.
What Sajal Labs should NOT reinvent: A cross-vendor Vulkan GPU compute backend, or ARM NEON kernel tuning.
Source: github.com/Tencent/ncnn README.md

## TFLite / LiteRT (Google)

Problem: Give Google products and third-party devs a high-performance, low-latency, privacy-preserving on-device ML runtime across phones, embedded/IoT, web, desktop — now expanding into on-device GenAI.
Approach: Rebranded "LiteRT" (2024) — "the next generation of the world's most widely deployed ML runtime." Models convert to a compact `.tflite` FlatBuffer; runtime executes via legacy Interpreter API or newer CompiledModel API (introduced 2025, production Jan 2026) adding automated hardware selection and async execution. Hardware acceleration via delegates (GPU, NNAPI, Core ML, vendor NPU delegates).
Language: C++ core; Kotlin, Java, Swift, Python, JavaScript (LiteRT.js, July 2026, runs `.tflite` in-browser via WASM) bindings.
Model formats: Native `.tflite`; conversion from TensorFlow, PyTorch (incl. GenAI models), JAX.
CPU: yes, XNNPACK delegate as default optimized CPU backend. GPU: GPU delegate for mobile/desktop. Edge/mobile: core purpose — Android, iOS, embedded/IoT, now also web (WASM/WebGPU/WebNN) and desktop.
Quantization: Extensive — dynamic-range, full-integer, float16, int16-activation (16x8) PTQ, custom quantization specs.
Graph compilation: CompiledModel API explicitly compiles against selected hardware at/ahead of load time; Interpreter API interprets the FlatBuffer directly.
Serving: Pure on-device library across platforms; no HTTP server (not its purpose).
Python dependency: No, generally — Python API exists for desktop/tooling; native C++/Kotlin/Swift/JS remove Python for production mobile/embedded/web.
API surface: C++ core; Kotlin/Java, Swift, Python, JavaScript.

Strength: Almost certainly the most widely deployed on-device ML runtime in the world, years of production hardening, mature delegate ecosystem, active 2026 investment (CompiledModel API, LiteRT.js) — not a stagnant target.
Weakness: `.tflite` format + dual-API transition (Interpreter → CompiledModel) adds surface area/migration complexity; requires conversion from training framework like MNN/ncnn.
What Sajal Labs could learn: The delegate pattern (pluggable hardware backends behind one graph executor, mirroring ONNX Runtime's EPs) is repeatedly-validated architecture. Google is actively evolving this in 2026 — not a stagnant target to differentiate against.
What Sajal Labs should NOT reinvent: A general mobile-GPU/NPU delegate ecosystem, or a from-scratch quantization scheme.
Source: developers.googleblog.com/{litert-the-universal-framework-for-on-device-ai, tensorflow-lite-is-now-litert}; developers.google.com/edge/litert; github.com/google-ai-edge/litert

## LibTorch (PyTorch C++ API)

Problem: Let developers use PyTorch's tensor ops and trained models directly inside C++ apps/production systems without a Python runtime dependency.
Approach: PyTorch's C++ distribution — same underlying libtorch/ATen/c10 libraries backing the Python API, packaged as a downloadable ZIP (headers, static/shared libs, CMake config) a C++ project links against.
Language: C++ (mirrors much of Python `torch` namespace: `torch::Tensor`, `torch::nn`).
Model formats: Historically TorchScript (serializable/traceable representation from a Python model); PyTorch 2.x shifting toward `torch.export()`-based AOT flows (feeding AOTInductor/ExecuTorch) — official cppdocs fetched didn't fully detail the newer export path within LibTorch itself (marked partially unknown).
CPU: yes, CPU-only build variant. GPU: yes, CUDA-enabled build variant. Edge/mobile: unknown from primary sources fetched — LibTorch's distribution targets desktop/server; mobile PyTorch deployment now lives under ExecuTorch, per PyTorch's own project separation.
Quantization: not detailed in the cppdocs page fetched — PyTorch quantization APIs exist framework-wide but not confirmed specifically for the LibTorch distribution page.
Graph compilation: Not inherent to LibTorch itself — capture/compilation (TorchScript trace/script, or torch.export/AOTInductor) is a separate step producing an artifact LibTorch loads/executes.
Serving: Library only. TorchServe (the first-party HTTP/gRPC server) was archived by maintainers Aug 7, 2025 — "Limited Maintenance," no planned updates/fixes/security patches. Official guidance now points to Triton's LibTorch backend or vLLM (for LLMs).
Python dependency: No — LibTorch's entire purpose is running PyTorch computation in C++ without Python.
API surface: C++ only.

Strength: Direct access to the exact same tensor/autograd/op implementations as PyTorch Python — no semantic drift between training and C++ inference; simple CMake linking.
Weakness: No actively maintained first-party serving story any more (TorchServe archived Aug 2025); not minimal — pulls in the full PyTorch op/tensor library, not a model-specific reduced set (that's moved to ExecuTorch); mobile/edge story now lives in a separate project.
What Sajal Labs could learn: PyTorch's own trajectory — splitting "full C++ tensor library for servers" (LibTorch) from "minimal edge runtime" (ExecuTorch) — is direct evidence that a minimal, model-specific runtime is a genuinely distinct problem from a general C++ tensor API.
What Sajal Labs should NOT reinvent: A general C++ autograd/tensor library matching Python-training numerics — if bit-for-bit PyTorch parity is ever required, link LibTorch rather than reimplementing ATen kernels.
Source: docs.pytorch.org/cppdocs/installing.html; github.com/pytorch/serve issues/3396, releases/tag/v0.10.0

## Cross-cutting observations (CPU/edge runtimes)

- **"Model-specific minimal runtime with only the operators a given model needs" is already solved, multiple times over** — ONNX Runtime's reduced-operator-config-file custom build, ExecuTorch's selective build (~50KB core, only linked kernels), and MNN's `MNN_BUILD_MINI` (fixed-shape, ~25% smaller) all do this today. A new project claiming this as its core differentiator must explain what it does differently, not just that it does it.
- **Python is uniformly optional at inference time, never actually required, across all 8 projects.** "No Python at inference" is table stakes here, not a differentiator.
- **No project here does true single-file, self-contained model+runtime distribution** in the llama.cpp/GGUF sense (one file, no separate SDK to link). ExecuTorch's `.pte` is closest — a single serialized artifact — but still requires linking the separate ExecuTorch runtime library. This is a plausibly underserved angle.
- **Two clear architectural families:** (1) proprietary/optimized-format runtimes optimized hard for size/dependency-freedom (ncnn, MNN, XNNPACK, ExecuTorch's `.pte`), vs. (2) standardized-interchange-format runtimes optimized for breadth/interop (ONNX Runtime, OpenVINO, TFLite/LiteRT). Most of the genuinely novel space is closer to family (1).
- **Kernel-level SIMD optimization is a shared, reused layer.** XNNPACK alone backs TFLite, PyTorch Mobile/ExecuTorch, ONNX Runtime, TensorFlow.js. A new minimal runtime should almost certainly embed an existing kernel library rather than hand-writing SIMD kernels.
- **First-party HTTP serving is the exception, not the rule.** Only OpenVINO (OVMS) ships an actively maintained official server; PyTorch's own TorchServe was archived Aug 2025. If "serving story" is where Sajal Labs differentiates, that's a legitimately open lane precisely because it's a hard, separate problem (batching, concurrency, multi-tenancy).
- **Quantization (INT8 minimum) is universal and mature** across all 8 — not a gap.
- **PyTorch itself split "general C++ tensor library" (LibTorch) from "minimal edge runtime" (ExecuTorch)** — direct evidence from a major framework that these are different engineering problems, supporting the premise that a dedicated minimal-artifact compiler is its own project, not a LibTorch feature request.
- **Hugging Face Hub-based single-artifact distribution already exists in nascent form** via `optimum-executorch` (select Transformer architectures → `.pte` from the Hub). Narrow, but prior art worth differentiating against explicitly.
- **"Edge/mobile" means different things per project** — OpenVINO's edge is industrial/PC edge; ncnn/MNN/TFLite/ExecuTorch's edge is phone/embedded/MCU. Any competitive framing should specify which segment Sajal Labs targets.
