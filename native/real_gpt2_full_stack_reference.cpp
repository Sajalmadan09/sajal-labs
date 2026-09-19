// exp31: hand-written reference for real GPT-2's FULL 12-layer decoder
// stack (hidden=768, 12 heads, ffn=3072), built independently of
// sajal_compile.py's codegen. Same GPT2Block math as exp30's
// real_gpt2_reference.cpp (pre-norm, combined Q/K/V via one buffer —
// exp29's addressing trick — causal masking, tanh-GELU), just looped
// over all 12 real layers instead of hand-copied twice.
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common.hpp"

struct GPT2Block {
    static constexpr int D = 768, D_FF = 3072, H = 12, DH = D / H;
    std::vector<float> qkv_weight, qkv_bias, o_weight, o_bias;
    std::vector<float> ln1_weight, ln1_bias, ff1_weight, ff1_bias, ff2_weight, ff2_bias, ln2_weight, ln2_bias;

    std::vector<float> forward(const std::vector<float>& X, int n) const {
        std::vector<float> x1 = X;
        layernorm_rows(x1.data(), n, D, ln1_weight.data(), ln1_bias.data(), 1e-5f);

        std::vector<float> QKV(static_cast<size_t>(n) * 3 * D);
        linear(x1.data(), n, D, qkv_weight.data(), qkv_bias.data(), 3 * D, QKV.data());

        const float scale = 1.0f / std::sqrt(static_cast<float>(DH));
        std::vector<float> scores(static_cast<size_t>(n) * n);
        std::vector<float> ctx(static_cast<size_t>(n) * D);
        for (int h = 0; h < H; ++h) {
            const float* Qh = QKV.data() + 0 * D + h * DH;
            const float* Kh = QKV.data() + 1 * D + h * DH;
            const float* Vh = QKV.data() + 2 * D + h * DH;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, DH, scale, Qh, 3 * D, Kh, 3 * D,
                        0.0f, scores.data(), n);
            causal_mask_rows(scores.data(), n);
            softmax_rows(scores.data(), n, n);
            float* ctx_h = ctx.data() + h * DH;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, DH, n, 1.0f, scores.data(),
                        n, Vh, 3 * D, 0.0f, ctx_h, D);
        }

        std::vector<float> attn_out(static_cast<size_t>(n) * D);
        linear(ctx.data(), n, D, o_weight.data(), o_bias.data(), D, attn_out.data());

        std::vector<float> x = X;
        add_inplace(x.data(), attn_out.data(), x.size());

        std::vector<float> x2 = x;
        layernorm_rows(x2.data(), n, D, ln2_weight.data(), ln2_bias.data(), 1e-5f);

        std::vector<float> h_(static_cast<size_t>(n) * D_FF);
        linear(x2.data(), n, D, ff1_weight.data(), ff1_bias.data(), D_FF, h_.data());
        gelu_tanh_inplace(h_.data(), h_.size());

        std::vector<float> ff_out(static_cast<size_t>(n) * D);
        linear(h_.data(), n, D_FF, ff2_weight.data(), ff2_bias.data(), D, ff_out.data());

        add_inplace(x.data(), ff_out.data(), x.size());
        return x;
    }
};

GPT2Block load_block(const std::string& dir, const std::string& prefix) {
    GPT2Block m;
    m.qkv_weight = load_f32(dir + "/" + prefix + "attn_c_attn_weight.bin", static_cast<size_t>(3 * GPT2Block::D) * GPT2Block::D);
    m.qkv_bias = load_f32(dir + "/" + prefix + "attn_c_attn_bias.bin", 3 * GPT2Block::D);
    m.o_weight = load_f32(dir + "/" + prefix + "attn_c_proj_weight.bin", static_cast<size_t>(GPT2Block::D) * GPT2Block::D);
    m.o_bias = load_f32(dir + "/" + prefix + "attn_c_proj_bias.bin", GPT2Block::D);
    m.ln1_weight = load_f32(dir + "/" + prefix + "ln_1_weight.bin", GPT2Block::D);
    m.ln1_bias = load_f32(dir + "/" + prefix + "ln_1_bias.bin", GPT2Block::D);
    m.ff1_weight = load_f32(dir + "/" + prefix + "mlp_c_fc_weight.bin", static_cast<size_t>(GPT2Block::D_FF) * GPT2Block::D);
    m.ff1_bias = load_f32(dir + "/" + prefix + "mlp_c_fc_bias.bin", GPT2Block::D_FF);
    m.ff2_weight = load_f32(dir + "/" + prefix + "mlp_c_proj_weight.bin", static_cast<size_t>(GPT2Block::D) * GPT2Block::D_FF);
    m.ff2_bias = load_f32(dir + "/" + prefix + "mlp_c_proj_bias.bin", GPT2Block::D);
    m.ln2_weight = load_f32(dir + "/" + prefix + "ln_2_weight.bin", GPT2Block::D);
    m.ln2_bias = load_f32(dir + "/" + prefix + "ln_2_bias.bin", GPT2Block::D);
    return m;
}

int main(int argc, char** argv) {
    if (argc != 4 || std::string(argv[2]) != "run") {
        std::cerr << "usage: " << argv[0] << " <dir> run <n_layers>\n";
        return 1;
    }
    std::string dir = argv[1];
    int n_layers = std::stoi(argv[3]);
    std::vector<GPT2Block> blocks;
    for (int i = 0; i < n_layers; ++i) {
        blocks.push_back(load_block(dir, "layers_" + std::to_string(i) + "_"));
    }
    std::ifstream cfg(dir + "/test_config.txt");
    int in_dim, n_test;
    cfg >> in_dim >> n_test;
    auto X = load_f32(dir + "/test_inputs.bin", static_cast<size_t>(n_test) * in_dim);
    for (const auto& block : blocks) {
        X = block.forward(X, n_test);
    }
    std::ofstream out(dir + "/handwritten_outputs.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(X.data()), X.size() * sizeof(float));
    std::cerr << "wrote handwritten_outputs.bin\n";
    return 0;
}
