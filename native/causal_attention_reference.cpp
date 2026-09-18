// exp27: hand-written reference for causal (autoregressive) multi-head
// self-attention, built independently of sajal_compile.py's codegen —
// same per-head BLAS-slicing pattern as native/multihead_reference.cpp
// (exp20), plus common.hpp's causal_mask_rows() applied to each head's
// raw scores before softmax_rows().
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common.hpp"

struct CausalMultiHeadAttention {
    static constexpr int D = 16, H = 4, DH = D / H;
    std::vector<float> q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, o_weight, o_bias;

    std::vector<float> forward(const std::vector<float>& X, int n) const {
        std::vector<float> Q(static_cast<size_t>(n) * D), K(Q.size()), V(Q.size());
        linear(X.data(), n, D, q_weight.data(), q_bias.data(), D, Q.data());
        linear(X.data(), n, D, k_weight.data(), k_bias.data(), D, K.data());
        linear(X.data(), n, D, v_weight.data(), v_bias.data(), D, V.data());

        const float scale = 1.0f / std::sqrt(static_cast<float>(DH));
        std::vector<float> scores(static_cast<size_t>(n) * n);
        std::vector<float> ctx(Q.size());
        for (int h = 0; h < H; ++h) {
            const float* Qh = Q.data() + h * DH;
            const float* Kh = K.data() + h * DH;
            const float* Vh = V.data() + h * DH;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, DH, scale, Qh, D, Kh, D,
                        0.0f, scores.data(), n);
            causal_mask_rows(scores.data(), n);
            softmax_rows(scores.data(), n, n);
            float* ctx_h = ctx.data() + h * DH;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, DH, n, 1.0f, scores.data(),
                        n, Vh, D, 0.0f, ctx_h, D);
        }

        std::vector<float> out(Q.size());
        linear(ctx.data(), n, D, o_weight.data(), o_bias.data(), D, out.data());
        return out;
    }
};

CausalMultiHeadAttention load_model(const std::string& dir) {
    CausalMultiHeadAttention m;
    m.q_weight = load_f32(dir + "/q_weight.bin", static_cast<size_t>(CausalMultiHeadAttention::D) * CausalMultiHeadAttention::D);
    m.q_bias = load_f32(dir + "/q_bias.bin", CausalMultiHeadAttention::D);
    m.k_weight = load_f32(dir + "/k_weight.bin", static_cast<size_t>(CausalMultiHeadAttention::D) * CausalMultiHeadAttention::D);
    m.k_bias = load_f32(dir + "/k_bias.bin", CausalMultiHeadAttention::D);
    m.v_weight = load_f32(dir + "/v_weight.bin", static_cast<size_t>(CausalMultiHeadAttention::D) * CausalMultiHeadAttention::D);
    m.v_bias = load_f32(dir + "/v_bias.bin", CausalMultiHeadAttention::D);
    m.o_weight = load_f32(dir + "/o_weight.bin", static_cast<size_t>(CausalMultiHeadAttention::D) * CausalMultiHeadAttention::D);
    m.o_bias = load_f32(dir + "/o_bias.bin", CausalMultiHeadAttention::D);
    return m;
}

int main(int argc, char** argv) {
    if (argc != 3 || std::string(argv[2]) != "run") {
        std::cerr << "usage: " << argv[0] << " <dir> run\n";
        return 1;
    }
    std::string dir = argv[1];
    CausalMultiHeadAttention m = load_model(dir);
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
