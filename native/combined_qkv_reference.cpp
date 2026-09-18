// exp29: hand-written reference for multi-head attention with a
// COMBINED Q/K/V projection (one Linear producing 3*d outputs, sliced
// into Q/K/V — GPT-2's real c_attn shape, closing the gap exp28 named
// and deferred), built independently of sajal_compile.py's codegen.
// Reuses the exact per-head BLAS-slicing trick already validated since
// exp20, one level up: Q/K/V are column-slices of ONE [n, 3*d] buffer
// (offsets 0, d, 2*d; lda = 3*d), not three separate [n, d] buffers.
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common.hpp"

struct CombinedQKVAttention {
    static constexpr int D = 16, H = 4, DH = D / H;
    std::vector<float> qkv_weight, qkv_bias, o_weight, o_bias;

    std::vector<float> forward(const std::vector<float>& X, int n) const {
        std::vector<float> QKV(static_cast<size_t>(n) * 3 * D);
        linear(X.data(), n, D, qkv_weight.data(), qkv_bias.data(), 3 * D, QKV.data());

        const float scale = 1.0f / std::sqrt(static_cast<float>(DH));
        std::vector<float> scores(static_cast<size_t>(n) * n);
        std::vector<float> ctx(static_cast<size_t>(n) * D);
        for (int h = 0; h < H; ++h) {
            const float* Qh = QKV.data() + 0 * D + h * DH;
            const float* Kh = QKV.data() + 1 * D + h * DH;
            const float* Vh = QKV.data() + 2 * D + h * DH;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, DH, scale, Qh, 3 * D, Kh, 3 * D,
                        0.0f, scores.data(), n);
            softmax_rows(scores.data(), n, n);
            float* ctx_h = ctx.data() + h * DH;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, DH, n, 1.0f, scores.data(),
                        n, Vh, 3 * D, 0.0f, ctx_h, D);
        }

        std::vector<float> out(static_cast<size_t>(n) * D);
        linear(ctx.data(), n, D, o_weight.data(), o_bias.data(), D, out.data());
        return out;
    }
};

CombinedQKVAttention load_model(const std::string& dir) {
    CombinedQKVAttention m;
    m.qkv_weight = load_f32(dir + "/qkv_weight.bin", static_cast<size_t>(3 * CombinedQKVAttention::D) * CombinedQKVAttention::D);
    m.qkv_bias = load_f32(dir + "/qkv_bias.bin", 3 * CombinedQKVAttention::D);
    m.o_weight = load_f32(dir + "/o_weight.bin", static_cast<size_t>(CombinedQKVAttention::D) * CombinedQKVAttention::D);
    m.o_bias = load_f32(dir + "/o_bias.bin", CombinedQKVAttention::D);
    return m;
}

int main(int argc, char** argv) {
    if (argc != 3 || std::string(argv[2]) != "run") {
        std::cerr << "usage: " << argv[0] << " <dir> run\n";
        return 1;
    }
    std::string dir = argv[1];
    CombinedQKVAttention m = load_model(dir);
    std::ifstream cfg(dir + "/test_config.txt");
    int in_dim, n_test;
    cfg >> in_dim >> n_test;
    auto X = load_f32(dir + "/test_inputs.bin", static_cast<size_t>(n_test) * in_dim);
    auto Y = m.forward(X, n_test);
    std::ofstream out(dir + "/handwritten_outputs.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(Y.data()), Y.size() * sizeof(float));
    std::cerr << "wrote handwritten_outputs.bin\n";
    return 0;
}
