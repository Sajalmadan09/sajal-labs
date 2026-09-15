// Native reimplementation of python/tiny_transformer.py's forward pass —
// see transformer_model.hpp for the TransformerBlock struct itself and the
// full explanation of the attention BLAS trick. This file is just the
// run/bench/once driver used by exp2-8's benchmark scripts, kept
// deliberately unchanged by the CLI work so those experiments' numbers
// stay reproducible.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "transformer_model.hpp"

void run_mode(const std::string& artifacts_dir) {
    auto t0 = Clock::now();
    TransformerBlock m = load_transformer(artifacts_dir);
    std::ifstream shapes(artifacts_dir + "/shapes.txt");
    int seq_len, d_model, n_heads, d_ff, n_test;
    shapes >> seq_len >> d_model >> n_heads >> d_ff >> n_test;

    auto X_all = load_f32(artifacts_dir + "/test_inputs.bin",
                           static_cast<size_t>(n_test) * seq_len * d_model);

    std::vector<float> outputs(static_cast<size_t>(n_test) * seq_len * d_model);
    std::vector<float> x(static_cast<size_t>(seq_len) * d_model);  // reused per-test-vector input buffer
    for (int t = 0; t < n_test; ++t) {
        std::copy(X_all.begin() + static_cast<size_t>(t) * seq_len * d_model,
                  X_all.begin() + static_cast<size_t>(t + 1) * seq_len * d_model, x.begin());
        const auto& y = m.forward(x);
        std::copy(y.begin(), y.end(), outputs.begin() + static_cast<size_t>(t) * seq_len * d_model);
    }

    std::ofstream out(artifacts_dir + "/native_outputs.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(outputs.data()), outputs.size() * sizeof(float));

    auto ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    std::cerr << "wrote native_outputs.bin (" << n_test << "x" << seq_len << "x" << d_model
              << ") in " << ms << " ms\n";
}

void bench_mode(const std::string& artifacts_dir) {
    auto t_start = Clock::now();
    TransformerBlock m = load_transformer(artifacts_dir);
    auto cold_start_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();

    const int WARMUP = 50, ITERS = 500;
    std::vector<float> x(static_cast<size_t>(m.seq_len) * m.d_model, 0.1f);  // batch size 1

    for (int i = 0; i < WARMUP; ++i) m.forward(x);

    std::vector<double> latencies_ms;
    latencies_ms.reserve(ITERS);
    for (int i = 0; i < ITERS; ++i) {
        auto t0 = Clock::now();
        const auto& y = m.forward(x);
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

// exp7: one cold invocation, one inference, exit — see mlp.cpp's once_mode
// for why this exists and why it deliberately does no internal timing.
void once_mode(const std::string& artifacts_dir) {
    TransformerBlock m = load_transformer(artifacts_dir);
    std::vector<float> x(static_cast<size_t>(m.seq_len) * m.d_model, 0.1f);
    const auto& y = m.forward(x);
    asm volatile("" : : "g"(y.data()) : "memory");
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
