// exp28: hand-written reference for a real GPT-2 decoder block (2
// layers, hidden=768, 12 heads, ffn=3072), built independently of
// sajal_compile.py's codegen. Pre-norm (LayerNorm before the sub-block,
// GPT-2's real ordering, the OPPOSITE of every prior reference's
// post-norm), causal masking (common.hpp's causal_mask_rows, exp27), and
// tanh-approximation GELU (common.hpp's gelu_tanh_inplace, exp28) — all
// three genuinely new to this reference file, not reused from an earlier
// exp's reference unchanged.
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common.hpp"

struct GPT2Block {
    static constexpr int D = 768, D_FF = 3072, H = 12, DH = D / H;
    std::vector<float> q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, o_weight, o_bias;
    std::vector<float> ln1_weight, ln1_bias, ff1_weight, ff1_bias, ff2_weight, ff2_bias, ln2_weight, ln2_bias;

    std::vector<float> forward(const std::vector<float>& X, int n) const {
        std::vector<float> x1 = X;
        layernorm_rows(x1.data(), n, D, ln1_weight.data(), ln1_bias.data(), 1e-5f);

        std::vector<float> Q(static_cast<size_t>(n) * D), K(Q.size()), V(Q.size());
        linear(x1.data(), n, D, q_weight.data(), q_bias.data(), D, Q.data());
        linear(x1.data(), n, D, k_weight.data(), k_bias.data(), D, K.data());
        linear(x1.data(), n, D, v_weight.data(), v_bias.data(), D, V.data());

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

        std::vector<float> attn_out(Q.size());
        linear(ctx.data(), n, D, o_weight.data(), o_bias.data(), D, attn_out.data());

        std::vector<float> x = X;
        add_inplace(x.data(), attn_out.data(), x.size());

        std::vector<float> x2 = x;
        layernorm_rows(x2.data(), n, D, ln2_weight.data(), ln2_bias.data(), 1e-5f);

        std::vector<float> h_(static_cast<size_t>(n) * D_FF);
        linear(x2.data(), n, D, ff1_weight.data(), ff1_bias.data(), D_FF, h_.data());
        gelu_tanh_inplace(h_.data(), h_.size());

        std::vector<float> ff_out(Q.size());
        linear(h_.data(), n, D_FF, ff2_weight.data(), ff2_bias.data(), D, ff_out.data());

        add_inplace(x.data(), ff_out.data(), x.size());
        return x;
    }
};

GPT2Block load_block(const std::string& dir, const std::string& prefix) {
    GPT2Block m;
    m.q_weight = load_f32(dir + "/" + prefix + "q_weight.bin", static_cast<size_t>(GPT2Block::D) * GPT2Block::D);
    m.q_bias = load_f32(dir + "/" + prefix + "q_bias.bin", GPT2Block::D);
    m.k_weight = load_f32(dir + "/" + prefix + "k_weight.bin", static_cast<size_t>(GPT2Block::D) * GPT2Block::D);
    m.k_bias = load_f32(dir + "/" + prefix + "k_bias.bin", GPT2Block::D);
    m.v_weight = load_f32(dir + "/" + prefix + "v_weight.bin", static_cast<size_t>(GPT2Block::D) * GPT2Block::D);
    m.v_bias = load_f32(dir + "/" + prefix + "v_bias.bin", GPT2Block::D);
    m.o_weight = load_f32(dir + "/" + prefix + "o_weight.bin", static_cast<size_t>(GPT2Block::D) * GPT2Block::D);
    m.o_bias = load_f32(dir + "/" + prefix + "o_bias.bin", GPT2Block::D);
    m.ln1_weight = load_f32(dir + "/" + prefix + "ln1_weight.bin", GPT2Block::D);
    m.ln1_bias = load_f32(dir + "/" + prefix + "ln1_bias.bin", GPT2Block::D);
    m.ff1_weight = load_f32(dir + "/" + prefix + "ff1_weight.bin", static_cast<size_t>(GPT2Block::D_FF) * GPT2Block::D);
    m.ff1_bias = load_f32(dir + "/" + prefix + "ff1_bias.bin", GPT2Block::D_FF);
    m.ff2_weight = load_f32(dir + "/" + prefix + "ff2_weight.bin", static_cast<size_t>(GPT2Block::D) * GPT2Block::D_FF);
    m.ff2_bias = load_f32(dir + "/" + prefix + "ff2_bias.bin", GPT2Block::D);
    m.ln2_weight = load_f32(dir + "/" + prefix + "ln2_weight.bin", GPT2Block::D);
    m.ln2_bias = load_f32(dir + "/" + prefix + "ln2_bias.bin", GPT2Block::D);
    return m;
}

int main(int argc, char** argv) {
    if (argc != 3 || std::string(argv[2]) != "run") {
        std::cerr << "usage: " << argv[0] << " <dir> run\n";
        return 1;
    }
    std::string dir = argv[1];
    GPT2Block block0 = load_block(dir, "blocks_0_");
    GPT2Block block1 = load_block(dir, "blocks_1_");
    std::ifstream cfg(dir + "/test_config.txt");
    int in_dim, n_test;
    cfg >> in_dim >> n_test;
    auto X = load_f32(dir + "/test_inputs.bin", static_cast<size_t>(n_test) * in_dim);
    auto Y = block1.forward(block0.forward(X, n_test), n_test);
    std::ofstream out(dir + "/handwritten_outputs.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(Y.data()), Y.size() * sizeof(float));
    std::cerr << "wrote handwritten_outputs.bin\n";
    return 0;
}
