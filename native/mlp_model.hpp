// The MLP model struct + loader, factored out of mlp.cpp so both the
// dedicated mlp.cpp driver (kept as-is for exp1/exp9/10's reproducible
// benchmark numbers) and the sajal CLI (which loads models directly,
// in-process, rather than shelling out to a second binary) can share it.
#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"

struct MLP {
    int in_dim, hidden_dim, out_dim;
    std::vector<float> W1, b1, W2, b2;  // W1:[hidden,in]  W2:[out,hidden]

    // X:[n,in] row-major -> returns Y:[n,out] row-major, softmax applied per row.
    // Built from the shared op library in common.hpp — see its comments for
    // why W's [out,in] layout needs no physical transpose, and why softmax
    // subtracts the row max first.
    std::vector<float> forward(const std::vector<float>& X, int n) const {
        std::vector<float> H(static_cast<size_t>(n) * hidden_dim);
        linear(X.data(), n, in_dim, W1.data(), b1.data(), hidden_dim, H.data());
        relu_inplace(H.data(), H.size());

        std::vector<float> Y(static_cast<size_t>(n) * out_dim);
        linear(H.data(), n, hidden_dim, W2.data(), b2.data(), out_dim, Y.data());
        softmax_rows(Y.data(), n, out_dim);
        return Y;
    }
};

inline MLP load_mlp(const std::string& artifacts_dir) {
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
