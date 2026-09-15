# Sajal Labs

Native AI inference research: can trained ML models be transformed into
minimal, dependency-light C/C++ inference artifacts without losing
predictive fidelity — and when does that actually beat Python-based serving?

**Philosophy:** train anywhere, deploy natively, measure everything. No
claim that C++ is inherently faster — every advantage is a hypothesis to
benchmark, not an assumption.

## Status

Phase 0 — landscape research (not started). Per the project brief, no
runtime/compiler code gets written until the existing ecosystem (ONNX
Runtime, llama.cpp, ExecuTorch, TVM, etc.) is surveyed and a real gap is
identified.

## Roadmap

0. Landscape research & gap analysis
1. Educational prototype (tiny NN in Python/C/C++)
2. Native runtime core (Tensor, Linear, ReLU, Softmax)
3. Model equivalence (real trained model)
4. Benchmarking vs. strong baselines
5. Packaging
6. Developer experience (CLI)
7. Hugging Face integration
8. Transformer support
9. Hardware optimization (SIMD/CUDA)
10. Research publication
