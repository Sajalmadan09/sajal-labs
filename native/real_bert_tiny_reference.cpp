// exp25: hand-written reference for prajjwal1/bert-tiny's real 2-layer
// encoder (hidden=128, heads=2, intermediate=512, eps=1e-12), built
// independently of sajal_compile.py's codegen — same shape as
// stacked_blocks_reference.cpp (exp23), just real dims/weights instead of
// exp21's toy d=16. Not the same code path as native/bert_model.hpp
// (exp11) — that one also does token embeddings and is designed for
// variable-length real sentences; this is scoped to exactly what
// export_real_bert_tiny_test.py's rewired wrapper computes: the layer
// stack alone, given already-embedded input.
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common.hpp"

struct RewiredBertLayer {
    static constexpr int D = 128, D_FF = 512, H = 2, DH = D / H;
    std::vector<float> q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, o_weight, o_bias;
    std::vector<float> ln1_weight, ln1_bias, ff1_weight, ff1_bias, ff2_weight, ff2_bias, ln2_weight, ln2_bias;

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
            softmax_rows(scores.data(), n, n);
            float* ctx_h = ctx.data() + h * DH;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, DH, n, 1.0f, scores.data(),
                        n, Vh, D, 0.0f, ctx_h, D);
        }

        std::vector<float> attn_out(Q.size());
        linear(ctx.data(), n, D, o_weight.data(), o_bias.data(), D, attn_out.data());

        std::vector<float> x1 = X;
        add_inplace(x1.data(), attn_out.data(), x1.size());
        layernorm_rows(x1.data(), n, D, ln1_weight.data(), ln1_bias.data(), 1e-12f);

        std::vector<float> h_(static_cast<size_t>(n) * D_FF);
        linear(x1.data(), n, D, ff1_weight.data(), ff1_bias.data(), D_FF, h_.data());
        gelu_inplace(h_.data(), h_.size());

        std::vector<float> ff_out(Q.size());
        linear(h_.data(), n, D_FF, ff2_weight.data(), ff2_bias.data(), D, ff_out.data());

        std::vector<float> x2 = x1;
        add_inplace(x2.data(), ff_out.data(), x2.size());
        layernorm_rows(x2.data(), n, D, ln2_weight.data(), ln2_bias.data(), 1e-12f);
        return x2;
    }
};

RewiredBertLayer load_layer(const std::string& dir, const std::string& prefix) {
    RewiredBertLayer m;
    m.q_weight = load_f32(dir + "/" + prefix + "q_weight.bin", static_cast<size_t>(RewiredBertLayer::D) * RewiredBertLayer::D);
    m.q_bias = load_f32(dir + "/" + prefix + "q_bias.bin", RewiredBertLayer::D);
    m.k_weight = load_f32(dir + "/" + prefix + "k_weight.bin", static_cast<size_t>(RewiredBertLayer::D) * RewiredBertLayer::D);
    m.k_bias = load_f32(dir + "/" + prefix + "k_bias.bin", RewiredBertLayer::D);
    m.v_weight = load_f32(dir + "/" + prefix + "v_weight.bin", static_cast<size_t>(RewiredBertLayer::D) * RewiredBertLayer::D);
    m.v_bias = load_f32(dir + "/" + prefix + "v_bias.bin", RewiredBertLayer::D);
    m.o_weight = load_f32(dir + "/" + prefix + "o_weight.bin", static_cast<size_t>(RewiredBertLayer::D) * RewiredBertLayer::D);
    m.o_bias = load_f32(dir + "/" + prefix + "o_bias.bin", RewiredBertLayer::D);
    m.ln1_weight = load_f32(dir + "/" + prefix + "ln1_weight.bin", RewiredBertLayer::D);
    m.ln1_bias = load_f32(dir + "/" + prefix + "ln1_bias.bin", RewiredBertLayer::D);
    m.ff1_weight = load_f32(dir + "/" + prefix + "ff1_weight.bin", static_cast<size_t>(RewiredBertLayer::D_FF) * RewiredBertLayer::D);
    m.ff1_bias = load_f32(dir + "/" + prefix + "ff1_bias.bin", RewiredBertLayer::D_FF);
    m.ff2_weight = load_f32(dir + "/" + prefix + "ff2_weight.bin", static_cast<size_t>(RewiredBertLayer::D) * RewiredBertLayer::D_FF);
    m.ff2_bias = load_f32(dir + "/" + prefix + "ff2_bias.bin", RewiredBertLayer::D);
    m.ln2_weight = load_f32(dir + "/" + prefix + "ln2_weight.bin", RewiredBertLayer::D);
    m.ln2_bias = load_f32(dir + "/" + prefix + "ln2_bias.bin", RewiredBertLayer::D);
    return m;
}

int main(int argc, char** argv) {
    if (argc != 3 || std::string(argv[2]) != "run") {
        std::cerr << "usage: " << argv[0] << " <dir> run\n";
        return 1;
    }
    std::string dir = argv[1];
    RewiredBertLayer layer0 = load_layer(dir, "layers_0_");
    RewiredBertLayer layer1 = load_layer(dir, "layers_1_");
    std::ifstream cfg(dir + "/test_config.txt");
    int in_dim, n_test;
    cfg >> in_dim >> n_test;
    auto X = load_f32(dir + "/test_inputs.bin", static_cast<size_t>(n_test) * in_dim);
    auto Y = layer1.forward(layer0.forward(X, n_test), n_test);
    std::ofstream out(dir + "/handwritten_outputs.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(Y.data()), Y.size() * sizeof(float));
    std::cerr << "wrote handwritten_outputs.bin\n";
    return 0;
}
