// exp16: hand-written reference for the Linear->LayerNorm->GELU->Linear->
// Softmax test architecture, built independently of sajal_compile.py's
// codegen, to give the compiler's output a second ground truth beyond the
// PyTorch reference — same validation bar exp15 held (bit-identical, not
// just numerically close).
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common.hpp"

struct LNGeluNet {
    static constexpr int IN_DIM = 32, HIDDEN_DIM = 64, OUT_DIM = 10;
    std::vector<float> fc1_weight, fc1_bias, ln_weight, ln_bias, fc2_weight, fc2_bias;

    std::vector<float> forward(const std::vector<float>& X, int n) const {
        std::vector<float> h(static_cast<size_t>(n) * HIDDEN_DIM);
        linear(X.data(), n, IN_DIM, fc1_weight.data(), fc1_bias.data(), HIDDEN_DIM, h.data());
        layernorm_rows(h.data(), n, HIDDEN_DIM, ln_weight.data(), ln_bias.data(), 1e-5f);
        gelu_inplace(h.data(), h.size());

        std::vector<float> y(static_cast<size_t>(n) * OUT_DIM);
        linear(h.data(), n, HIDDEN_DIM, fc2_weight.data(), fc2_bias.data(), OUT_DIM, y.data());
        softmax_rows(y.data(), n, OUT_DIM);
        return y;
    }
};

LNGeluNet load_model(const std::string& dir) {
    LNGeluNet m;
    m.fc1_weight = load_f32(dir + "/fc1_weight.bin", static_cast<size_t>(LNGeluNet::HIDDEN_DIM) * LNGeluNet::IN_DIM);
    m.fc1_bias = load_f32(dir + "/fc1_bias.bin", LNGeluNet::HIDDEN_DIM);
    m.ln_weight = load_f32(dir + "/ln_weight.bin", LNGeluNet::HIDDEN_DIM);
    m.ln_bias = load_f32(dir + "/ln_bias.bin", LNGeluNet::HIDDEN_DIM);
    m.fc2_weight = load_f32(dir + "/fc2_weight.bin", static_cast<size_t>(LNGeluNet::OUT_DIM) * LNGeluNet::HIDDEN_DIM);
    m.fc2_bias = load_f32(dir + "/fc2_bias.bin", LNGeluNet::OUT_DIM);
    return m;
}

int main(int argc, char** argv) {
    if (argc != 3 || std::string(argv[2]) != "run") {
        std::cerr << "usage: " << argv[0] << " <dir> run\n";
        return 1;
    }
    std::string dir = argv[1];
    LNGeluNet m = load_model(dir);
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
