# Development Environment (as of 2026-09-15)

| | |
|---|---|
| Machine | Mac mini |
| OS | macOS 26.3 (Darwin 25.3.0), arm64 |
| CPU | Apple M4, 10 cores |
| RAM | 16 GB |
| GPU | Apple M4 integrated GPU (Metal) — **no NVIDIA/CUDA GPU** |
| Compiler | Apple clang 17.0.0 (Xcode CLT at /Library/Developer/CommandLineTools) |
| CMake | not installed |
| Python | 3.13.2 (miniconda, `/opt/homebrew/Caskroom/miniconda/base`) |
| Relevant libs installed | none (no OpenBLAS/oneDNN/Eigen/LibTorch/ONNX Runtime/llama.cpp via brew) |

## Implications for the project (per brief §12, §34)

- **CPU benchmarking**: ARM64 / Apple Silicon only. No x86 comparison unless run on other hardware later.
- **GPU benchmarking**: Metal/Apple Silicon GPU only. **CUDA, cuBLAS, CUTLASS, TensorRT cannot be benchmarked on this machine** — research them for the literature review, but any performance claim about them must come from their own published numbers, cited as such, not measured by us.
- **Toolchain to install before Phase 1 code**: CMake, and later whichever of OpenBLAS/Accelerate/oneDNN we decide to link against (Accelerate is preinstalled system framework on macOS, no install needed).
- Apple's **Accelerate framework** (BLAS/vDSP/BNNS) is available out of the box and is the natural first native math baseline on this hardware — worth prioritizing in research over CUDA-only libraries.
