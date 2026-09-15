// Native reimplementation of python/tiny_transformer.py's forward pass:
// multi-head self-attention (residual + LayerNorm) followed by a GELU
// feed-forward (residual + LayerNorm). Built entirely from common.hpp's
// shared op library except for the attention head-splitting itself, which
// uses a BLAS trick worth understanding rather than hiding: Q/K/V are each
// one [seq,d_model] matrix, and a "head" is just a column-slice of it. In
// row-major storage that slice isn't contiguous — but cblas_sgemm's lda/ldb
// parameters mean "stride between rows in the underlying buffer", which can
// be wider than the columns actually used. So each head's [seq,d_head]
// submatrix is addressed directly (pointer + d_model as the stride) with
// zero copying, instead of physically splitting Q/K/V into per-head arrays.
//
// Scratch buffers (Q, K, V, ctx, ...) are allocated ONCE in allocate_buffers()
// and reused across every forward() call, rather than freshly heap-allocated
// per call. exp3 (research/experiments/exp3-size-sweep/results.md) found
// native C++ lost its warm-latency edge over ONNX Runtime at every tested
// transformer size, and named per-call allocation (~10 std::vectors per
// forward() vs. ONNX Runtime's reused memory arena) as the likely cause,
// not tested — this file is that test.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common.hpp"

struct TransformerBlock {
    int seq_len, d_model, n_heads, d_ff, d_head;
    std::vector<float> Wq, bq, Wk, bk, Wv, bv, Wo, bo;
    std::vector<float> ln1_w, ln1_b;
    std::vector<float> W1, b1, W2, b2;
    std::vector<float> ln2_w, ln2_b;

    // Scratch space, sized once by allocate_buffers() after shapes are known.
    // `mutable` because forward() stays logically const (same weights, same
    // math) even though it writes into this reused scratch space — the
    // mutability is an implementation detail (a cache), not part of the
    // object's observable state.
    mutable std::vector<float> Q, K, V, ctx, scores, attn_out, x1, ff_hidden, ff_out, x2;

    void allocate_buffers() {
        size_t SD = static_cast<size_t>(seq_len) * d_model;
        Q.resize(SD);
        K.resize(SD);
        V.resize(SD);
        ctx.resize(SD);
        scores.resize(static_cast<size_t>(seq_len) * seq_len);
        attn_out.resize(SD);
        x1.resize(SD);
        ff_hidden.resize(static_cast<size_t>(seq_len) * d_ff);
        ff_out.resize(SD);
        x2.resize(SD);
    }

    // X:[seq_len,d_model] row-major -> returns a reference to this object's
    // internal x2 buffer (valid until the next forward() call — the caller
    // must copy it out before calling forward() again, same contract as any
    // reused scratch buffer / object pool).
    const std::vector<float>& forward(const std::vector<float>& X) const {
        const int S = seq_len, D = d_model, H = n_heads, Dh = d_head, F = d_ff;

        linear(X.data(), S, D, Wq.data(), bq.data(), D, Q.data());
        linear(X.data(), S, D, Wk.data(), bk.data(), D, K.data());
        linear(X.data(), S, D, Wv.data(), bv.data(), D, V.data());

        const float scale = 1.0f / std::sqrt(static_cast<float>(Dh));
        for (int h = 0; h < H; ++h) {
            const float* Qh = Q.data() + h * Dh;
            const float* Kh = K.data() + h * Dh;
            const float* Vh = V.data() + h * Dh;
            // scores = (Qh @ Kh^T) * scale.  Qh, Kh are [S,Dh] slices with row
            // stride D (that's what the trailing "D" args are — see file comment).
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, S, S, Dh, scale, Qh, D, Kh, D,
                        0.0f, scores.data(), S);
            softmax_rows(scores.data(), S, S);
            // ctx[:, h] = scores @ Vh — write straight into this head's column
            // slice of ctx (ldc=D) instead of a temporary that gets copied in.
            float* ctx_h = ctx.data() + h * Dh;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, S, Dh, S, 1.0f, scores.data(),
                        S, Vh, D, 0.0f, ctx_h, D);
        }

        linear(ctx.data(), S, D, Wo.data(), bo.data(), D, attn_out.data());

        for (size_t i = 0; i < x1.size(); ++i) x1[i] = X[i] + attn_out[i];  // residual
        layernorm_rows(x1.data(), S, D, ln1_w.data(), ln1_b.data());

        linear(x1.data(), S, D, W1.data(), b1.data(), F, ff_hidden.data());
        gelu_inplace(ff_hidden.data(), ff_hidden.size());

        linear(ff_hidden.data(), S, F, W2.data(), b2.data(), D, ff_out.data());

        for (size_t i = 0; i < x2.size(); ++i) x2[i] = x1[i] + ff_out[i];  // residual
        layernorm_rows(x2.data(), S, D, ln2_w.data(), ln2_b.data());
        return x2;
    }
};

TransformerBlock load_model(const std::string& artifacts_dir) {
    std::ifstream shapes(artifacts_dir + "/shapes.txt");
    if (!shapes) throw std::runtime_error("cannot open shapes.txt");
    TransformerBlock m;
    int n_test;
    shapes >> m.seq_len >> m.d_model >> m.n_heads >> m.d_ff >> n_test;
    m.d_head = m.d_model / m.n_heads;

    auto load_lin = [&](const std::string& name, int out_dim, int in_dim, std::vector<float>& W,
                         std::vector<float>& b) {
        W = load_f32(artifacts_dir + "/" + name + "_weight.bin", static_cast<size_t>(out_dim) * in_dim);
        b = load_f32(artifacts_dir + "/" + name + "_bias.bin", out_dim);
    };
    load_lin("q_proj", m.d_model, m.d_model, m.Wq, m.bq);
    load_lin("k_proj", m.d_model, m.d_model, m.Wk, m.bk);
    load_lin("v_proj", m.d_model, m.d_model, m.Wv, m.bv);
    load_lin("out_proj", m.d_model, m.d_model, m.Wo, m.bo);
    load_lin("ff1", m.d_ff, m.d_model, m.W1, m.b1);
    load_lin("ff2", m.d_model, m.d_ff, m.W2, m.b2);
    m.ln1_w = load_f32(artifacts_dir + "/ln1_weight.bin", m.d_model);
    m.ln1_b = load_f32(artifacts_dir + "/ln1_bias.bin", m.d_model);
    m.ln2_w = load_f32(artifacts_dir + "/ln2_weight.bin", m.d_model);
    m.ln2_b = load_f32(artifacts_dir + "/ln2_bias.bin", m.d_model);
    m.allocate_buffers();
    return m;
}

void run_mode(const std::string& artifacts_dir) {
    auto t0 = Clock::now();
    TransformerBlock m = load_model(artifacts_dir);
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
    TransformerBlock m = load_model(artifacts_dir);
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
