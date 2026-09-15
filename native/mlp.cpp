// Native reimplementation of python/tiny_mlp.py's forward pass:
//   softmax(relu(x @ W1^T + b1) @ W2^T + b2)
//
// The matmuls are delegated to Apple's Accelerate framework (cblas_sgemm) —
// per research/related-work.md, hand-writing a competitive GEMM kernel is
// exactly the kind of work an existing math library already solves, and
// Accelerate is the only public path to the AMX matrix coprocessor on this
// Apple Silicon machine. ReLU and softmax are trivial enough to hand-write.
//
// The MLP struct itself lives in mlp_model.hpp (shared with the sajal CLI);
// this file is just the run/bench/once driver used by exp1/exp9/10's
// benchmark scripts, kept deliberately unchanged by the CLI work so those
// experiments' numbers stay reproducible.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mlp_model.hpp"

void run_mode(const std::string& artifacts_dir) {
    auto t0 = Clock::now();
    MLP m = load_mlp(artifacts_dir);
    std::ifstream shapes(artifacts_dir + "/shapes.txt");
    int in_dim, hidden_dim, out_dim, n_test;
    shapes >> in_dim >> hidden_dim >> out_dim >> n_test;

    auto X = load_f32(artifacts_dir + "/test_inputs.bin", static_cast<size_t>(n_test) * in_dim);
    auto Y = m.forward(X, n_test);

    std::ofstream out(artifacts_dir + "/native_outputs.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(Y.data()), Y.size() * sizeof(float));

    auto ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    std::cerr << "wrote native_outputs.bin (" << n_test << "x" << out_dim << ") in " << ms << " ms\n";
}

void bench_mode(const std::string& artifacts_dir) {
    auto t_start = Clock::now();
    MLP m = load_mlp(artifacts_dir);
    auto cold_start_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();

    const int WARMUP = 50, ITERS = 500;
    std::vector<float> x(m.in_dim, 0.1f);  // fixed single input vector, batch size 1

    for (int i = 0; i < WARMUP; ++i) m.forward(x, 1);

    std::vector<double> latencies_ms;
    latencies_ms.reserve(ITERS);
    for (int i = 0; i < ITERS; ++i) {
        auto t0 = Clock::now();
        auto y = m.forward(x, 1);
        latencies_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        asm volatile("" : : "g"(y.data()) : "memory");  // prevent the optimizer from eliding the call
    }

    double mean = 0;
    for (double v : latencies_ms) mean += v;
    mean /= latencies_ms.size();

    std::printf(
        "{\"impl\": \"native_cpp\", \"cold_start_ms\": %.4f, \"warmup_iters\": %d, \"iters\": %d, "
        "\"mean_ms\": %.5f, \"p50_ms\": %.5f, \"p90_ms\": %.5f, \"p95_ms\": %.5f, \"p99_ms\": %.5f}\n",
        cold_start_ms, WARMUP, ITERS, mean, percentile(latencies_ms, 50), percentile(latencies_ms, 90),
        percentile(latencies_ms, 95), percentile(latencies_ms, 99));
}

// exp7: one cold invocation, one inference, exit — no warmup, no loop, no
// output file. This is what a serverless/CLI/edge caller actually pays:
// process start -> model ready -> one answer -> process teardown. The
// orchestrator times the WHOLE subprocess from outside, so this mode
// deliberately does no internal timing or printing of its own.
void once_mode(const std::string& artifacts_dir) {
    MLP m = load_mlp(artifacts_dir);
    std::vector<float> x(m.in_dim, 0.1f);
    auto y = m.forward(x, 1);
    asm volatile("" : : "g"(y.data()) : "memory");  // prevent the optimizer from eliding the call
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <artifacts_dir> <run|bench|once>\n";
        return 1;
    }
    std::string artifacts_dir = argv[1], mode = argv[2];
    try {
        if (mode == "run") {
            run_mode(artifacts_dir);
        } else if (mode == "bench") {
            bench_mode(artifacts_dir);
        } else if (mode == "once") {
            once_mode(artifacts_dir);
        } else {
            std::cerr << "unknown mode: " << mode << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
