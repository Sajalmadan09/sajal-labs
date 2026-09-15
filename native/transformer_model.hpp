// The TransformerBlock struct + loader, factored out of transformer.cpp so
// both the dedicated transformer.cpp driver (kept as-is for exp2-8's
// reproducible benchmark numbers) and the sajal CLI can share it.
//
// Attention uses per-head cblas_sgemm calls (Q/K/V column-slices addressed
// via BLAS's lda/ldb stride trick, no physical splitting) — see
// transformer.cpp's original header comment for the full explanation, and
// research/experiments/exp5-6 for why this beats the BLAS-free alternative.
#pragma once

#include <fstream>
#include <stdexcept>
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
    // math) even though it writes into this reused scratch space.
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
    // internal x2 buffer (valid until the next forward() call).
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
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, S, S, Dh, scale, Qh, D, Kh, D,
                        0.0f, scores.data(), S);
            softmax_rows(scores.data(), S, S);
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

inline TransformerBlock load_transformer(const std::string& artifacts_dir) {
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
