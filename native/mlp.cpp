// Native reimplementation of python/tiny_mlp.py's forward pass:
//   softmax(relu(x @ W1^T + b1) @ W2^T + b2)
//
// The matmuls are delegated to Apple's Accelerate framework (cblas_sgemm) —
// per research/related-work.md, hand-writing a competitive GEMM kernel is
// exactly the kind of work an existing math library already solves, and
// Accelerate is the only public path to the AMX matrix coprocessor on this
// Apple Silicon machine. ReLU and softmax are trivial enough to hand-write.
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

std::vector<float> load_f32(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<float> data(count);
    f.read(reinterpret_cast<char*>(data.data()), count * sizeof(float));
    if (!f) throw std::runtime_error("short read: " + path);
    return data;
}

struct MLP {
    int in_dim, hidden_dim, out_dim;
    std::vector<float> W1, b1, W2, b2;  // W1:[hidden,in]  W2:[out,hidden]

    // X:[n,in] row-major -> returns Y:[n,out] row-major, softmax applied per row.
    std::vector<float> forward(const std::vector<float>& X, int n) const {
        std::vector<float> H(static_cast<size_t>(n) * hidden_dim);
        // H = X * W1^T.  W1 is stored [hidden,in] (PyTorch's nn.Linear layout),
        // so CblasTrans on W1 gives exactly the [in,hidden] operand we need —
        // no physical transpose required, BLAS does it via the trans flag.
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, hidden_dim, in_dim,
                    1.0f, X.data(), in_dim, W1.data(), in_dim, 0.0f, H.data(), hidden_dim);
        for (int i = 0; i < n; ++i) {
            float* row = H.data() + static_cast<size_t>(i) * hidden_dim;
            for (int j = 0; j < hidden_dim; ++j) {
                float v = row[j] + b1[j];
                row[j] = v > 0.0f ? v : 0.0f;  // ReLU
            }
        }

        std::vector<float> Y(static_cast<size_t>(n) * out_dim);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, out_dim, hidden_dim,
                    1.0f, H.data(), hidden_dim, W2.data(), hidden_dim, 0.0f, Y.data(), out_dim);
        for (int i = 0; i < n; ++i) {
            float* row = Y.data() + static_cast<size_t>(i) * out_dim;
            for (int j = 0; j < out_dim; ++j) row[j] += b2[j];
            // Softmax, subtracting the row max first for numerical stability
            // (avoids overflowing exp() on large logits) — standard practice,
            // not something PyTorch's softmax skips either.
            float mx = *std::max_element(row, row + out_dim);
            float sum = 0.0f;
            for (int j = 0; j < out_dim; ++j) {
                row[j] = std::exp(row[j] - mx);
                sum += row[j];
            }
            for (int j = 0; j < out_dim; ++j) row[j] /= sum;
        }
        return Y;
    }
};

MLP load_model(const std::string& artifacts_dir) {
    std::ifstream shapes(artifacts_dir + "/shapes.txt");
    if (!shapes) throw std::runtime_error("cannot open shapes.txt");
    MLP m;
    int n_test;
    shapes >> m.in_dim >> m.hidden_dim >> m.out_dim >> n_test;

    m.W1 = load_f32(artifacts_dir + "/fc1_weight.bin", static_cast<size_t>(m.hidden_dim) * m.in_dim);
    m.b1 = load_f32(artifacts_dir + "/fc1_bias.bin", m.hidden_dim);
    m.W2 = load_f32(artifacts_dir + "/fc2_weight.bin", static_cast<size_t>(m.out_dim) * m.hidden_dim);
    m.b2 = load_f32(artifacts_dir + "/fc2_bias.bin", m.out_dim);
    return m;
}

double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return v[lo];
    double frac = idx - lo;
    return v[lo] * (1 - frac) + v[hi] * frac;  // matches numpy.percentile's default 'linear' method
}

void run_mode(const std::string& artifacts_dir) {
    auto t0 = Clock::now();
    MLP m = load_model(artifacts_dir);
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
    MLP m = load_model(artifacts_dir);
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

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <artifacts_dir> <run|bench>\n";
        return 1;
    }
    std::string artifacts_dir = argv[1], mode = argv[2];
    try {
        if (mode == "run") {
            run_mode(artifacts_dir);
        } else if (mode == "bench") {
            bench_mode(artifacts_dir);
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
