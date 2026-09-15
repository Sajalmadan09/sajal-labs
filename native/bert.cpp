// exp11 driver: run/bench/once over TinyBert (see bert_model.hpp). Mirrors
// mlp.cpp/transformer.cpp's mode conventions.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "bert_model.hpp"

std::vector<int> load_i32(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<int32_t> raw(count);
    f.read(reinterpret_cast<char*>(raw.data()), count * sizeof(int32_t));
    if (!f) throw std::runtime_error("short read: " + path);
    return std::vector<int>(raw.begin(), raw.end());
}

// Returns each test case's real sequence length, parsed from test_cases.txt
// ("<len> <sentence text>" per line) — written by python/bert_export.py.
std::vector<int> load_case_lengths(const std::string& artifacts_dir) {
    std::ifstream f(artifacts_dir + "/test_cases.txt");
    if (!f) throw std::runtime_error("cannot open test_cases.txt");
    std::vector<int> lengths;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        int len;
        iss >> len;
        lengths.push_back(len);
    }
    return lengths;
}

void run_mode(const std::string& artifacts_dir) {
    auto t0 = Clock::now();
    TinyBert m = load_bert(artifacts_dir);
    auto lengths = load_case_lengths(artifacts_dir);

    for (size_t i = 0; i < lengths.size(); ++i) {
        std::string prefix = artifacts_dir + "/case" + std::to_string(i) + "_";
        auto token_ids = load_i32(prefix + "input_ids.bin", lengths[i]);
        auto out = m.forward(token_ids);

        std::ofstream hout(prefix + "native_hidden.bin", std::ios::binary);
        hout.write(reinterpret_cast<char*>(out.hidden.data()), out.hidden.size() * sizeof(float));
        std::ofstream pout(prefix + "native_pooled.bin", std::ios::binary);
        pout.write(reinterpret_cast<char*>(out.pooled.data()), out.pooled.size() * sizeof(float));
    }

    auto ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    std::cerr << "wrote native_hidden/pooled.bin for " << lengths.size() << " cases in " << ms << " ms\n";
}

void bench_mode(const std::string& artifacts_dir) {
    auto t_start = Clock::now();
    TinyBert m = load_bert(artifacts_dir);
    auto cold_start_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();

    auto lengths = load_case_lengths(artifacts_dir);
    const int BENCH_CASE = 1;  // "The quick brown fox..." — 12 tokens, representative short sentence
    auto token_ids = load_i32(artifacts_dir + "/case" + std::to_string(BENCH_CASE) + "_input_ids.bin",
                               lengths[BENCH_CASE]);

    const int WARMUP = 50, ITERS = 500;
    for (int i = 0; i < WARMUP; ++i) m.forward(token_ids);

    std::vector<double> latencies_ms;
    latencies_ms.reserve(ITERS);
    for (int i = 0; i < ITERS; ++i) {
        auto t0 = Clock::now();
        auto out = m.forward(token_ids);
        latencies_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        asm volatile("" : : "g"(out.pooled.data()) : "memory");
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

void once_mode(const std::string& artifacts_dir) {
    TinyBert m = load_bert(artifacts_dir);
    auto lengths = load_case_lengths(artifacts_dir);
    const int BENCH_CASE = 1;
    auto token_ids = load_i32(artifacts_dir + "/case" + std::to_string(BENCH_CASE) + "_input_ids.bin",
                               lengths[BENCH_CASE]);
    auto out = m.forward(token_ids);
    asm volatile("" : : "g"(out.pooled.data()) : "memory");
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
