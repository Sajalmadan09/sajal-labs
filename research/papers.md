# Academic Literature Survey

Scope note: primary sources only (arXiv/conference/technical report pages). Where a "paper" is really vendor documentation rather than peer-reviewed work (this happens once, for XLA), that is stated explicitly rather than dressed up as an academic citation. This is a targeted survey via search, not an exhaustive systematic review — see the completeness caveat at the end.

## 1. Model Quantization (PTQ and QAT)

### Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (Jacob, Kligys, Chen, Zhu, Tang, Howard, Adam, Kalenichenko — CVPR 2018)
Problem: Floating-point inference is too slow/heavy for mobile CPUs; naive post-hoc quantization degrades accuracy.
Approach: A quantization scheme (affine, per-channel) plus a co-designed training procedure (fake-quant nodes simulating int8 arithmetic during training) letting inference run with integer-only arithmetic.
Key technical idea: Simulating quantization effects during the forward pass at training time (QAT) so the network learns weights robust to eventual int8 rounding, combined with an affine int8 scheme mapping cleanly onto integer-only kernels (no float ops at inference).
What Sajal Labs can learn/reuse: The affine quantization math (scale/zero-point, requantization after matmul/conv) is the reference spec to replicate bit-for-bit in a C/C++ int8 kernel — effectively "ground truth" for int8 inference semantics.
What Sajal Labs should NOT redo from scratch: Re-deriving quantization theory or re-running QAT experiments — standardized, implemented in TFLite/ONNX Runtime/QNNPACK; just implement the documented arithmetic.
Potential open research gap: Rounding-mode and accumulator-width choices in a from-scratch C/C++ kernel can silently diverge from the reference runtime's int8 output even when "using the same scheme" — bit-exactness is under-specified in the paper itself.
Source: https://arxiv.org/abs/1712.05877

### A Survey of Quantization Methods for Efficient Neural Network Inference (Gholami, Kim, Dong, Yao, Mahoney, Keutzer — arXiv 2021)
Problem: Fragmented literature (PTQ vs QAT, per-tensor vs per-channel, symmetric vs asymmetric, mixed precision) with no unifying reference.
Approach: Structured survey of uniform/non-uniform quantization, static vs dynamic, PTQ vs QAT trade-offs, hardware-specific considerations.
Key technical idea: Frames quantization error as a range/precision trade-off, catalogs which techniques recover accuracy at 8-bit vs 4-bit vs binary.
What Sajal Labs can learn/reuse: A map of the design space — likely target per-channel static int8 first, the best-supported/most portable choice for a hand-written C/C++ backend.
What Sajal Labs should NOT redo from scratch: Novel low-bit (sub-8-bit) quantization algorithms — active research, not to reinvent inside a compilation tool.
Potential open research gap: The survey evaluates quantization by accuracy-on-task, not by numerical agreement with a specific reference float implementation — the "equivalence to one designated reference run" framing is largely absent.
Source: https://arxiv.org/abs/2103.13630

## 2. Neural Network Pruning

### Learning both Weights and Connections for Efficient Neural Networks (Han, Pool, Tran, Dally — NeurIPS 2015)
Problem: Trained networks carry far more parameters than needed, inflating deployment memory/compute.
Approach: Train dense network → prune low-magnitude weights → retrain (fine-tune) the sparse network to recover accuracy.
Key technical idea: Iterative magnitude-based pruning + retraining recovers accuracy lost from pruning — 9x (AlexNet) and 13x (VGG-16) parameter reduction with no accuracy loss.
What Sajal Labs can learn/reuse: If Sajal Labs ever shrinks a model before compiling, magnitude pruning + fine-tune is the well-established, low-risk baseline.
What Sajal Labs should NOT redo from scratch: The retraining/fine-tuning loop belongs in the training framework, not the C/C++ inference compiler — pruning is upstream of "compile to native code."
Potential open research gap: Sparse weight storage in a hand-rolled C/C++ kernel trades code complexity for memory/latency wins; the paper doesn't address preserving bit-exact output when porting a sparsely-pruned model's kernels to a different runtime.
Source: https://arxiv.org/abs/1506.02626

### The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks (Frankle, Carbin — ICLR 2019)
Problem: Why do some pruned subnetworks train well from scratch while most don't?
Approach: Iterative magnitude pruning + rewinding to original initialization; empirically shows "winning ticket" subnetworks matching full-network accuracy.
Key technical idea: The specific initialization (not just sparse topology) matters — a pruned mask with its original initial weights trains successfully; the same mask with new random weights does not.
What Sajal Labs can learn/reuse: Mostly a training-dynamics result, not directly load-bearing for a static inference compiler working with already-trained models.
What Sajal Labs should NOT redo from scratch: The iterative pruning/rewinding search — expensive, training-time research, orthogonal to Sajal Labs' "no retraining" scope.
Potential open research gap: None directly — flagged for completeness; not a strong fit for a pure-inference project.
Source: https://arxiv.org/abs/1803.03635

## 3. Knowledge Distillation

### Distilling the Knowledge in a Neural Network (Hinton, Vinyals, Dean — arXiv 2015 / NeurIPS Deep Learning Workshop)
Problem: Large/ensemble models are accurate but too expensive to deploy — how to transfer their knowledge to a small, cheap model.
Approach: Train a smaller "student" to match a larger "teacher's" softened output distribution (temperature-scaled softmax) rather than only hard labels.
Key technical idea: Soft targets carry more information per example than one-hot labels (relative probabilities between wrong classes encode teacher "uncertainty"), enriching the student's training signal.
What Sajal Labs can learn/reuse: Training-time technique for a smaller architecture before compilation — relevant only if "shrink then compile" ever becomes a feature.
What Sajal Labs should NOT redo from scratch: The distillation training loop — squarely a training-framework concern, out of scope.
Potential open research gap: None specific to native-code compilation — upstream of Sajal Labs' stated scope.
Source: https://arxiv.org/abs/1503.02531

## 4. Operator / Kernel Fusion

### DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion (Niu, Guan, Wang, Agrawal, Ren, Shen, et al. — PLDI 2021)
Problem: Existing frameworks (TF, TVM, MNN) fuse operators using rigid, hand-written pattern lists that miss valid fusion opportunities in deep/diverse graphs, especially on mobile.
Approach: Classifies operators by mathematical/data-access properties, then uses graph rewriting + lightweight profiling to discover and generate fused kernels automatically, beyond fixed pattern matching.
Key technical idea: Fusion decisions driven by operator-property classification (not fixed templates) let the compiler fuse non-adjacent/unusual operator combinations, generating specialized fused code per graph.
What Sajal Labs can learn/reuse: Directly relevant blueprint for a C/C++ codegen backend — classify ops (elementwise, reduction, shuffling) and greedily fuse chains into single C loops to cut memory traffic and call overhead, exactly the mechanism producing "minimal, dependency-light" native code.
What Sajal Labs should NOT redo from scratch: Theorem-prover-verified substitution search (see TASO below) — heavyweight superoptimization; simple greedy pattern-based fusion (as DNNFusion does for mobile) is sufficient and much cheaper to implement first.
Potential open research gap: None of this literature asks "does the fused kernel produce numerically identical output to the unfused reference op-by-op execution" — fusion changes exact floating-point operation order (fused multiply-add vs. separate rounding), exactly the equivalence question Sajal Labs cares about and the fusion literature does not measure.
Source: https://arxiv.org/abs/2108.13342

### TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions (Jia, Padon, Thomas, Warszawski, Zaharia, Aiken — SOSP 2019)
Problem: Hand-written graph-optimization rules (TensorFlow has ~53,000 lines of them) are labor-intensive and incomplete.
Approach: Automatically generates candidate graph substitutions from operator specs, formally verifies each with an automated theorem prover, then does cost-based search to apply verified substitutions.
Key technical idea: Correctness-by-construction — every optimization (including fusions) is proven equivalent to the original subgraph via formal verification before firing, rather than trusted by convention.
What Sajal Labs can learn/reuse: Formally specifying operator semantics and checking equivalence is directly relevant to Sajal Labs' "preserve predictive behavior" goal — a template for reasoning rigorously about "does this rewrite change the answer."
What Sajal Labs should NOT redo from scratch: A general theorem-prover-backed substitution search is a multi-year research-systems effort — start with a small, manually-verified fusion rule set instead.
Potential open research gap: TASO's "equivalence" is exact at the symbolic/algebraic (real-number) level, not the floating-point level — it does not certify bit-for-bit or tolerance-bounded numerical equivalence after fusion, which is the actual guarantee Sajal Labs would need to make to users.
Source: https://theory.stanford.edu/~aiken/publications/papers/sosp19.pdf (also dl.acm.org/doi/10.1145/3341301.3359630)

## 5. ML Compilers / Graph Compilation

### TVM: An Automated End-to-End Optimizing Compiler for Deep Learning (Chen, Moreau, Jiang, Zheng, Yan, Shen, Cowan, Wang, Hu, Ceze, Guestrin, Krishnamurthy — OSDI 2018)
Problem: Deploying models efficiently across diverse hardware backends requires backend-specific hand-optimized kernels — not portable or automatable.
Approach: A compiler stack with graph-level optimizations (fusion, layout transforms) and operator-level scheduling, with a learned cost model auto-tuning low-level tensor programs per hardware target instead of hand-tuned kernels.
Key technical idea: Separating "what to compute" (tensor expression) from "how to schedule it" (loop order, tiling, vectorization), then ML-based search over the schedule space guided by a cost model, replacing manual kernel engineering.
What Sajal Labs can learn/reuse: The graph-IR → operator-IR → scheduled-code pipeline is the standard architecture for any "trained model to native code" compiler — even a much simpler compiler should mirror this layering rather than emitting code op-by-op naively.
What Sajal Labs should NOT redo from scratch: The auto-tuning/learned cost-model search infrastructure (AutoTVM/Ansor-style) — for a minimal C/C++ target, hand-picking good-enough loop schedules is far cheaper than reimplementing auto-tuning.
Potential open research gap: TVM optimizes for speed/portability, validating correctness by task accuracy or loose numerical tolerance — not "match this exact upstream framework's output to X ulps" as a first-class compiler contract, which is Sajal Labs' differentiator.
Source: https://arxiv.org/abs/1802.04799

### Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines (Ragan-Kelley, Barnes, Adams, Paris, Durand, Amarasinghe — PLDI 2013)
Problem: Image-processing pipelines require expert-tuned code for good performance; naive implementations are an order of magnitude slower.
Approach: Separates algorithm ("what") from schedule ("how" — order, tiling, fusion, parallelization), embedded as a C++ library/DSL, so the same algorithm can be retargeted with a different schedule.
Key technical idea: The algorithm/schedule separation this paper introduced is the conceptual ancestor of TVM's and XLA's own graph/schedule separation for tensor programs.
What Sajal Labs can learn/reuse: A foundational design pattern (not DL-specific) — the intellectual origin of "auto-schedule the low-level loops of a compiled kernel"; useful context, less directly actionable than TVM/DNNFusion.
What Sajal Labs should NOT redo from scratch: A full scheduling DSL/autoscheduler is out of scope — borrow the concept, not the machinery.
Potential open research gap: N/A directly (predates ML-specific equivalence concerns); included for provenance of the compiler techniques Sajal Labs leans on.
Source: https://dl.acm.org/doi/10.1145/2499370.2462176 (author copy: https://people.csail.mit.edu/jrk/halide-pldi13.pdf)

### XLA: Optimizing Compiler for Machine Learning (Google — technical documentation, NOT a peer-reviewed paper)
Note: no standalone peer-reviewed arXiv/conference paper found specifically for XLA — documented via Google's project docs/blog rather than an academic publication. Flagged rather than fabricating a citation.
Problem: TF/JAX/PyTorch graphs run as a sequence of pre-compiled kernel calls, leaving cross-op fusion and memory-traffic reduction on the table.
Approach: JIT/AOT-compiles a computation graph into fused, hardware-specific kernels (CPU/GPU/TPU), with operator fusion as its primary optimization.
Key technical idea: Whole-subgraph fusion into single generated kernels to minimize intermediate-tensor memory round-trips (reported ~7x speedup in an MLPerf BERT submission).
What Sajal Labs can learn/reuse: Concrete evidence fusion (likely Sajal Labs' biggest lever too) yields large real-world speedups — validates prioritizing a fusing C/C++ codegen over naive op-by-op emission.
What Sajal Labs should NOT redo from scratch: The full HLO IR and multi-backend infra — far beyond a minimal native-artifact tool's needs.
Potential open research gap: Same as TVM/DNNFusion — no published numerical-equivalence contract for AOT-compiled output vs. original eager execution.
Source: https://openxla.org/xla; https://github.com/openxla/xla

## 6. Low-Latency / Serving-Latency-Focused Inference Systems

### Clipper: A Low-Latency Online Prediction Serving System (Crankshaw, Wang, Zhou, Franklin, Gonzalez, Stoica — NSDI 2017)
Problem: ML frameworks handle training well but leave low-latency, robust, framework-agnostic serving as an afterthought.
Approach: A serving layer between applications and arbitrary ML frameworks adding caching, adaptive batching, and model selection to reduce tail latency while maintaining throughput.
Key technical idea: Framework-agnostic "container" abstraction for models plus adaptive batching trading a small latency budget for throughput, tuned per-request-load.
What Sajal Labs can learn/reuse: Adaptive batching and the "thin, fast, model-agnostic layer at the serving boundary" pattern is relevant if compiled artifacts get embedded in a serving loop — but Clipper solves a systems/scheduling problem, not compilation.
What Sajal Labs should NOT redo from scratch: A general multi-model serving/routing system — out of scope ("compile one model to one artifact" is Sajal Labs' scope).
Potential open research gap: Clipper measures latency/throughput, not whether served predictions numerically match a reference implementation — the equivalence axis is largely unaddressed.
Source: https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/crankshaw

### TensorFlow-Serving: Flexible, High-Performance ML Serving (Olston, Fiedel, Gorovoy, Harmsen, Lao, Li, Rajashekhar, Ramesh, Soyke — arXiv 2017)
Problem: Production-grade serving needs versioning, hot model swaps, consistent low latency at Google scale, beyond research-oriented serving scripts.
Approach: A C++ serving core with a plugin architecture for "servables" (a model type abstraction), batching scheduler, lifecycle management for versioned models.
Key technical idea: Decoupling the "servable" abstraction (any loadable, versioned prediction object, not just TF graphs) from the serving loop enables hot updates without downtime.
What Sajal Labs can learn/reuse: Precedent that a lean C++ core is the right choice for a serving/inference layer (validates going native); the servable/versioning abstraction is reusable if artifacts ever need in-place swapping.
What Sajal Labs should NOT redo from scratch: Full production serving infra (health checks, hot-swap versioning, multi-tenant batching) — orthogonal, mature, out of scope for a compilation tool.
Potential open research gap: Same gap as Clipper — no numerical-equivalence guarantee between serving output and a reference/training-time computation is discussed or measured.
Source: https://arxiv.org/abs/1712.06139

## 7. Edge AI / On-Device Inference

### MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (Howard, Zhu, Chen, Kalenichenko, Wang, Weyand, Andreetto, Adam — arXiv 2017)
Problem: Standard CNN architectures are too compute/memory-heavy for mobile/embedded deployment.
Approach: Replaces standard convolutions with depthwise-separable convolutions, plus two global hyperparameters (width/resolution multiplier) to trade accuracy for latency/size.
Key technical idea: Factorizing a standard convolution into depthwise (per-channel spatial filtering) + pointwise 1x1 drastically cuts multiply-add count with modest accuracy cost.
What Sajal Labs can learn/reuse: An architecture-level lesson — if users bring MobileNet-family models, the C/C++ codegen should have an efficient depthwise-conv kernel as a first-class primitive.
What Sajal Labs should NOT redo from scratch: Designing new efficient architectures — Sajal Labs compiles whatever architecture the user already trained.
Potential open research gap: None specific to compilation; useful mainly as "know your target workloads."
Source: https://arxiv.org/abs/1704.04861

### MCUNet: Tiny Deep Learning on IoT Devices (Lin, Chen, Lin, Gan, Han — NeurIPS 2020)
Problem: Running ImageNet-scale models on microcontrollers (KB-scale SRAM/flash, no OS) requires co-designing model and inference engine, not just shrinking the model.
Approach: Jointly designs a NAS-derived model (TinyNAS) and a matching lightweight inference engine (TinyEngine) with model-adaptive memory scheduling and kernel specialization.
Key technical idea: Memory scheduling specialized per-model at compile time (not a general dynamic allocator) plus per-layer kernel specialization — lower memory use and faster inference than generic engines (TF-Lite Micro, CMSIS-NN) on the same model.
What Sajal Labs can learn/reuse: **Directly the closest prior art to Sajal Labs' stated goal** — a compiler specializing memory layout and kernel code per specific trained model rather than using a generic runtime; TinyEngine's "model-adaptive" codegen is a strong architectural reference.
What Sajal Labs should NOT redo from scratch: Neural architecture search (TinyNAS) — a separate model-design problem; Sajal Labs takes the trained model as given.
Potential open research gap: MCUNet's TinyEngine optimizes for memory/latency on a known-accuracy model; it doesn't report/aim for numerical equivalence to a specific reference (e.g. PyTorch fp32) run — the "same model, compiled, does it produce the same numbers" question is unaddressed, and its target hardware (MCUs) generally can't even use the same float precision as the reference anyway.
Source: https://arxiv.org/abs/2007.10319 (github.com/mit-han-lab/mcunet; NeurIPS 2020 proceedings)

## 8. Numerical Reproducibility / Deterministic Inference in Floating Point

### What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg — ACM Computing Surveys, 1991)
Problem: Systems builders routinely misunderstand floating-point behavior, leading to subtle correctness bugs.
Approach: Tutorial covering IEEE-754 representation, rounding error, guard digits, catastrophic vs. benign cancellation.
Key technical idea: Formalizes why floating-point addition/multiplication are not associative and bounds the relative error (ULPs) introduced by rounding at each operation — the root mathematical cause of `(a+b)+c != a+(b+c)`.
What Sajal Labs can learn/reuse: **The foundational reference** for why a re-implemented C/C++ kernel that reorders operations (different loop order, SIMD width, fused-multiply-add vs. not) can produce a bit-different — though typically ULP-close — result from the original framework, even with "the same" math. Essential for defining what "numerical equivalence" should even mean (bit-exact vs. tolerance-bounded).
What Sajal Labs should NOT redo from scratch: The floating-point theory itself — settled, standardized (IEEE-754); use as citation/spec, don't re-derive.
Potential open research gap: Classic numerical analysis addresses generic FP summation, not the DNN-specific case of reduction order across fused conv/matmul kernels — translating this general theory into a concrete tolerance policy for a model-compilation tool is still a design (not solved-research) problem.
Source: https://dl.acm.org/doi/10.1145/103162.103163 (mirrored: docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html)

### Impacts of Floating-Point Non-Associativity on Reproducibility for HPC and Deep Learning Applications (Shanmugavelu, Taillefumier, Culver, Hernandez, et al. — SC'24 Workshops / CORRECTNESS 2024)
Problem: DL training/inference pipelines on GPUs are sometimes extremely sensitive to floating-point non-associativity, which can block certification, robustness assessment, and bug detection.
Approach: Empirically characterizes run-to-run variability caused by non-associative reductions in both classical HPC iterative algorithms and DL training/inference pipelines.
Key technical idea: "Non-reproducibility" in DL is not a training artifact alone — inference-time reductions (e.g. matmul/conv accumulation) inherit the same non-associativity issue as HPC codes, severe enough to fail correctness/regression testing.
What Sajal Labs can learn/reuse: Directly validates that Sajal Labs' "measure numerical/predictive equivalence" goal is a real, actively-studied problem, not solved — citable evidence for why bit-exactness across implementations should not be assumed even for "the same" model.
What Sajal Labs should NOT redo from scratch: General HPC reproducibility tooling/methodology — reuse their framing (variability magnitude vs. tolerance thresholds) rather than building new statistical infrastructure.
Potential open research gap: The paper studies existing frameworks' internal non-determinism (GPU kernel-to-kernel), not the specific case Sajal Labs cares about — equivalence between a hand-compiled single-threaded C/C++ CPU kernel and the original Python/framework reference. **That specific comparison looks like a genuine gap.**
Source: https://arxiv.org/abs/2408.05148

## 9. Model Equivalence / Behavioral Equivalence Across Conversion & Reimplementation

### An Empirical Study of Challenges in Converting Deep Learning Models (Openja, Nikanjam, Haj Yahmed, Khomh, Jiang — arXiv 2022 / IEEE conference)
Problem: No prior empirical evaluation existed of whether converting trained models (Keras/PyTorch to ONNX/CoreML) preserves prediction accuracy, performance, robustness.
Approach: Trains five widely-used DL models on three datasets in Keras and PyTorch, converts each to ONNX and CoreML, measures accuracy, latency, memory, model size before/after conversion across runtimes.
Key technical idea: Systematic, multi-model, multi-format empirical comparison protocol — same trained weights, different serialized/runtime representations, direct before/after measurement of predictive behavior.
What Sajal Labs can learn/reuse: **Close to the exact validation methodology Sajal Labs needs** for its own "did the C/C++ artifact preserve behavior" claim — reuse their evaluation protocol (accuracy parity + performance deltas across conversion) as a template.
What Sajal Labs should NOT redo from scratch: The general finding that most format conversions preserve accuracy "at the same level" is already established for ONNX/CoreML — no need to re-prove converters can work in general, just verify it for Sajal Labs' own compilation path.
Potential open research gap: This study evaluates format conversion (ONNX/CoreML), not compilation to raw, dependency-free C/C++ — a materially different transformation (no runtime, no standard op library, hand-written kernels) not subjected to the same empirical rigor. **That gap is squarely where Sajal Labs sits.**
Source: https://arxiv.org/abs/2206.14322

### Analysis of Failures and Risks in Deep Learning Model Converters: A Case Study in the ONNX Ecosystem (Louloudakis, Gibson, Rajan, Neubauer — arXiv 2023)
Problem: Model converters (PyTorch/TF to ONNX and back) are widely used but their failure modes are poorly characterized.
Approach: Examines ~200 real reported issues in ONNX-related PyTorch/TF converter repositories, categorizing failure stage and root cause.
Key technical idea: The node-conversion stage (mapping individual ops between frameworks) accounts for ~75% of defects, and ~33% of reported failures produce semantically/numerically incorrect (not just crashing) models — silent correctness bugs are common, not just crashes.
What Sajal Labs can learn/reuse: Strong evidence for where to focus correctness testing — per-operator translation is highest-risk, so prioritize exhaustive per-op numerical testing (each op in isolation, then compositions) over end-to-end-only testing.
What Sajal Labs should NOT redo from scratch: Re-cataloging generic converter bug taxonomies — reuse their failure taxonomy as a checklist for Sajal Labs' own op-by-op test suite.
Potential open research gap: The study covers existing converters translating between full-featured frameworks/runtimes (retaining generic tensor ops, broadcasting, etc.); it doesn't address the harder case of dropping to minimal, dependency-free native code where even basic runtime services (dynamic shapes, generic broadcasting, libm rounding) may not be available or may differ. **That specific failure surface is unexplored.**
Source: https://arxiv.org/abs/2303.17708

## Cross-cutting observations

- **Well-solved / mature**: quantization schemes (int8 PTQ/QAT math), pruning, and knowledge distillation are mature, standardized techniques with reference implementations in every major framework — treat as "consume, don't reinvent" building blocks, none of it core to the compilation problem itself.
- **Well-solved / mature**: the graph-compilation stack (graph IR → op fusion → scheduled/tuned low-level code) pioneered by Halide and formalized by TVM/XLA/TASO is a proven architecture. Sajal Labs' compiler should copy this layering rather than invent a new one — the open question isn't "how to structure a DL compiler" but "how to keep output minimal and dependency-free while doing so," which none of these systems optimize for (they optimize for speed/portability and happily pull in large runtimes).
- **Genuinely under-studied**: none of the fusion/compiler papers (DNNFusion, TASO, TVM, XLA) treat "does the optimized/fused kernel produce the same floating-point output as the unoptimized reference, to what tolerance" as a first-class, measured property. They verify algebraic/task-accuracy equivalence, not numerical equivalence under a specific tolerance policy — close to a genuine gap.
- **Genuinely under-studied**: the model-conversion-bug literature (Openja et al., Louloudakis et al.) is the closest existing work to Sajal Labs' equivalence-measurement goal, but studies conversion between full frameworks/runtimes (ONNX, CoreML) — not compilation down to a runtime-free C/C++ artifact with hand-written kernels. The specific failure surface of "no generic runtime services available" (dynamic shapes, broadcasting, libm rounding differences, accumulator-width choices) has essentially no dedicated empirical study.
- **Well-established but under-applied to this problem**: classic numerical analysis (Goldberg 1991) fully explains why summation-order/operation-order changes shift results, and recent DL-specific reproducibility work (Shanmugavelu et al. 2024) confirms this matters in practice for DL pipelines — but nobody has published a concrete "tolerance policy for comparing a native-compiled CPU kernel against a Python/framework reference" methodology. **This looks like the single most actionable open gap for Sajal Labs to define and publish on.**
- **Edge AI research (MobileNets, MCUNet)** targets model efficiency and runtime co-design for constrained hardware, and MCUNet's TinyEngine is the closest prior art architecturally (per-model specialized memory scheduling/kernels) — but optimizes for a fixed, known-accuracy model on microcontrollers, never asking whether compiled output matches a specific fp32 reference bit-for-bit or within a stated tolerance.
- **Serving-latency systems (Clipper, TensorFlow-Serving)** solve a different, adjacent problem — routing/batching/versioning around a model already running inside some framework's runtime — rather than replacing that runtime with a native artifact. Useful for later "how do users deploy the compiled artifact" thinking, not for the compilation/equivalence problem itself.
- **Overall framing**: the literature strongly suggests the "compile trained model to minimal native code" pipeline (graph opt → fusion → codegen) is well-trodden, and "does format conversion preserve accuracy" has been empirically studied for existing runtimes — but the intersection of the two ("compile to a dependency-free artifact, and rigorously measure/report numerical equivalence at a defined tolerance against the original") does not appear to have a dedicated paper. That intersection is where Sajal Labs' novelty claim would need to live, and should be validated as a genuine gap before committing to it as the project's differentiator.

**Completeness caveat**: this is a targeted-search survey, not an exhaustive systematic review — it's plausible closely related papers exist (e.g. in MLSys, ASPLOS, or industry tech reports) not surfaced here. Before finalizing a research direction, a follow-up pass specifically searching MLSys/ASPLOS proceedings directly for "model equivalence" / "numerical fidelity" / "inference determinism" would be worthwhile, since those venues are underrepresented above.
