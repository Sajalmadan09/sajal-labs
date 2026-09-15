// Microbenchmark for exp5: does cblas_sgemm have a fixed per-call dispatch
// cost independent of matrix size, and does that explain why the
// transformer's 14-GEMM-call forward pass (exp3/exp4) lost its advantage
// over ONNX Runtime that the MLP's 2-GEMM-call forward pass (exp1) had?
//
// No PyTorch/ONNX comparison here and no correctness check — this
// experiment is purely about Accelerate's own call-dispatch cost in
// isolation, not about matching a reference implementation.
//
// Part A (size sweep): time one square MxKxN cblas_sgemm call, M=K=N
// swept from 1 to 512. If latency at the smallest sizes is far above
// zero, that floor IS (an estimate of) the fixed per-call cost — actual
// compute for a 1x1x1 or 4x4x4 matmul is nanoseconds.
//
// Part B (call count, same total work): pick two (M,K,N) shapes matching
// our real workloads, and for k in {1,2,4,8,14,28}, compare "k separate
// cblas_sgemm calls of width N" against "1 cblas_sgemm call of width k*N"
// — same total FLOPs (M*K*(k*N)*2) either way, only the call count
// differs. If fixed dispatch cost matters, batching should win, and the
// gap should widen as k grows.
#include <cstdio>
#include <fstream>
#include <functional>
#include <vector>

#include "common.hpp"

constexpr int WARMUP = 50, ITERS = 200;

double bench_call(const std::function<void()>& call) {
    for (int i = 0; i < WARMUP; ++i) call();
    std::vector<double> latencies_ms;
    latencies_ms.reserve(ITERS);
    for (int i = 0; i < ITERS; ++i) {
        auto t0 = Clock::now();
        call();
        latencies_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    return percentile(latencies_ms, 50);
}

void bench_size_sweep(const std::string& out_path) {
    std::ofstream out(out_path);
    out << "size,p50_ms\n";
    for (int n : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}) {
        std::vector<float> A(static_cast<size_t>(n) * n, 0.1f), B(A.size(), 0.2f), C(A.size());
        double p50 = bench_call([&] {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, A.data(), n,
                        B.data(), n, 0.0f, C.data(), n);
        });
        out << n << "," << p50 << "\n";
        std::fprintf(stderr, "size=%4d  p50=%.5f ms\n", n, p50);
    }
}

void bench_call_count(const std::string& out_path) {
    std::ofstream out(out_path);
    out << "shape,k,mode,p50_ms\n";
    // (M,K,N) base shapes: one matching attention-head scale
    // (seq_len=32, d_head=64, seq_len=32), one matching a full linear
    // projection at exp2/3/4's d_model=256 scale (seq_len=32, 256, 256).
    struct Shape { int M, K, N; std::string label; };
    for (auto shape : {Shape{32, 64, 32, "attn_head"}, Shape{32, 256, 256, "linear_proj"}}) {
        std::vector<float> A(static_cast<size_t>(shape.M) * shape.K, 0.1f);
        for (int k : {1, 2, 4, 8, 14, 28}) {
            // k separate calls, each producing an [M,N] output.
            std::vector<std::vector<float>> Bs(k, std::vector<float>(static_cast<size_t>(shape.K) * shape.N, 0.2f));
            std::vector<std::vector<float>> Cs(k, std::vector<float>(static_cast<size_t>(shape.M) * shape.N));
            double p50_separate = bench_call([&] {
                for (int i = 0; i < k; ++i) {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, shape.M, shape.N, shape.K,
                                1.0f, A.data(), shape.K, Bs[i].data(), shape.N, 0.0f, Cs[i].data(),
                                shape.N);
                }
            });

            // 1 batched call: same A, one B widened to k*N columns, same total FLOPs.
            std::vector<float> B_wide(static_cast<size_t>(shape.K) * shape.N * k, 0.2f);
            std::vector<float> C_wide(static_cast<size_t>(shape.M) * shape.N * k);
            int N_wide = shape.N * k;
            double p50_batched = bench_call([&] {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, shape.M, N_wide, shape.K, 1.0f,
                            A.data(), shape.K, B_wide.data(), N_wide, 0.0f, C_wide.data(), N_wide);
            });

            out << shape.label << "," << k << ",separate," << p50_separate << "\n";
            out << shape.label << "," << k << ",batched," << p50_batched << "\n";
            std::fprintf(stderr, "%s k=%2d  separate=%.5f ms  batched=%.5f ms  (%.2fx)\n",
                         shape.label.c_str(), k, p50_separate, p50_batched, p50_separate / p50_batched);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <output_dir>\n", argv[0]);
        return 1;
    }
    std::string out_dir = argv[1];
    bench_size_sweep(out_dir + "/size_sweep.csv");
    bench_call_count(out_dir + "/call_count.csv");
    return 0;
}
