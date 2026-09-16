// exp17: hand-written reference for the residual FFN test architecture
// (y = LayerNorm(x + Linear(GELU(Linear(x)))), then a classification head),
// built independently of sajal_compile.py's codegen, to give the DAG-aware
// compiler's output a second ground truth beyond the PyTorch reference —
// same bar exp15/16 held.
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common.hpp"

struct ResidualFFN {
    static constexpr int D = 16, D_FF = 64, OUT_DIM = 10;
    std::vector<float> ff1_weight, ff1_bias, ff2_weight, ff2_bias;
    std::vector<float> ln_weight, ln_bias, head_weight, head_bias;

    std::vector<float> forward(const std::vector<float>& X, int n) const {
        std::vector<float> h(static_cast<size_t>(n) * D_FF);
        linear(X.data(), n, D, ff1_weight.data(), ff1_bias.data(), D_FF, h.data());
        gelu_inplace(h.data(), h.size());

        std::vector<float> ff_out(static_cast<size_t>(n) * D);
        linear(h.data(), n, D_FF, ff2_weight.data(), ff2_bias.data(), D, ff_out.data());

        std::vector<float> y = X;  // residual: copy x, then add the FFN branch
        add_inplace(y.data(), ff_out.data(), y.size());
        layernorm_rows(y.data(), n, D, ln_weight.data(), ln_bias.data(), 1e-5f);

        std::vector<float> out(static_cast<size_t>(n) * OUT_DIM);
        linear(y.data(), n, D, head_weight.data(), head_bias.data(), OUT_DIM, out.data());
        softmax_rows(out.data(), n, OUT_DIM);
        return out;
    }
};

ResidualFFN load_model(const std::string& dir) {
    ResidualFFN m;
    m.ff1_weight = load_f32(dir + "/ff1_weight.bin", static_cast<size_t>(ResidualFFN::D_FF) * ResidualFFN::D);
    m.ff1_bias = load_f32(dir + "/ff1_bias.bin", ResidualFFN::D_FF);
    m.ff2_weight = load_f32(dir + "/ff2_weight.bin", static_cast<size_t>(ResidualFFN::D) * ResidualFFN::D_FF);
    m.ff2_bias = load_f32(dir + "/ff2_bias.bin", ResidualFFN::D);
    m.ln_weight = load_f32(dir + "/ln_weight.bin", ResidualFFN::D);
    m.ln_bias = load_f32(dir + "/ln_bias.bin", ResidualFFN::D);
    m.head_weight = load_f32(dir + "/head_weight.bin", static_cast<size_t>(ResidualFFN::OUT_DIM) * ResidualFFN::D);
    m.head_bias = load_f32(dir + "/head_bias.bin", ResidualFFN::OUT_DIM);
    return m;
}

int main(int argc, char** argv) {
    if (argc != 3 || std::string(argv[2]) != "run") {
        std::cerr << "usage: " << argv[0] << " <dir> run\n";
        return 1;
    }
    std::string dir = argv[1];
    ResidualFFN m = load_model(dir);
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
