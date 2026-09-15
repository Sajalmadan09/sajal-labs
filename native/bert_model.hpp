// exp11: a real pretrained BERT encoder (prajjwal1/bert-tiny — 2 layers,
// hidden=128, 2 heads). A fresh implementation rather than reusing
// transformer_model.hpp's TransformerBlock: that struct's scratch buffers
// are preallocated once at a FIXED size (exp4's optimization for exp2-8's
// synthetic benchmark), but real sentences have different lengths, so this
// sizes its buffers per forward() call instead — correctness for real,
// variable-length input first; exp3-6's dispatch-overhead lessons apply
// here too but aren't re-optimized yet (see exp11's results for why).
//
// Structurally this mirrors TransformerBlock's post-norm attention+FFN
// block (per-head cblas_sgemm calls via BLAS's lda/ldb stride trick — see
// transformer_model.hpp's comments) but adds what a real encoder needs on
// top: token/position/segment embeddings, an embedding LayerNorm, and N
// stacked layers instead of one.
#pragma once

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"

struct BertLayerWeights {
    std::vector<float> Wq, bq, Wk, bk, Wv, bv;
    std::vector<float> Wo, bo, attn_ln_w, attn_ln_b;
    std::vector<float> W1, b1, W2, b2, ff_ln_w, ff_ln_b;
};

struct BertOutput {
    std::vector<float> hidden;  // [seq_len, hidden_size], row-major
    std::vector<float> pooled;  // [hidden_size] — tanh(dense(hidden[CLS]))
};

struct TinyBert {
    int num_layers, hidden_size, num_heads, intermediate_size, vocab_size, max_position_embeddings, head_dim;
    float layer_norm_eps;
    std::vector<float> word_emb, pos_emb, type_emb, emb_ln_w, emb_ln_b;
    std::vector<BertLayerWeights> layers;
    std::vector<float> pooler_w, pooler_b;

    // token_ids: this call's actual sequence (any length) — no padding, no
    // attention mask needed since every position is real for an unpadded
    // single sentence processed alone.
    BertOutput forward(const std::vector<int>& token_ids) const {
        const int S = static_cast<int>(token_ids.size()), H = hidden_size;

        std::vector<float> x(static_cast<size_t>(S) * H);
        for (int i = 0; i < S; ++i) {
            int tid = token_ids[i];
            for (int d = 0; d < H; ++d) {
                x[static_cast<size_t>(i) * H + d] =
                    word_emb[static_cast<size_t>(tid) * H + d] + pos_emb[static_cast<size_t>(i) * H + d] +
                    type_emb[d];  // token_type_id 0 for every position (single-sentence input)
            }
        }
        layernorm_rows(x.data(), S, H, emb_ln_w.data(), emb_ln_b.data(), layer_norm_eps);

        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        for (const auto& L : layers) {
            std::vector<float> Q(static_cast<size_t>(S) * H), K(Q.size()), V(Q.size()), ctx(Q.size());
            linear(x.data(), S, H, L.Wq.data(), L.bq.data(), H, Q.data());
            linear(x.data(), S, H, L.Wk.data(), L.bk.data(), H, K.data());
            linear(x.data(), S, H, L.Wv.data(), L.bv.data(), H, V.data());

            std::vector<float> scores(static_cast<size_t>(S) * S);
            for (int h = 0; h < num_heads; ++h) {
                const float* Qh = Q.data() + h * head_dim;
                const float* Kh = K.data() + h * head_dim;
                const float* Vh = V.data() + h * head_dim;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, S, S, head_dim, scale, Qh, H, Kh, H,
                            0.0f, scores.data(), S);
                softmax_rows(scores.data(), S, S);
                float* ctx_h = ctx.data() + h * head_dim;
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, S, head_dim, S, 1.0f, scores.data(),
                            S, Vh, H, 0.0f, ctx_h, H);
            }

            std::vector<float> attn_out(Q.size());
            linear(ctx.data(), S, H, L.Wo.data(), L.bo.data(), H, attn_out.data());
            for (size_t i = 0; i < x.size(); ++i) x[i] += attn_out[i];  // residual
            layernorm_rows(x.data(), S, H, L.attn_ln_w.data(), L.attn_ln_b.data(), layer_norm_eps);

            std::vector<float> ff_hidden(static_cast<size_t>(S) * intermediate_size);
            linear(x.data(), S, H, L.W1.data(), L.b1.data(), intermediate_size, ff_hidden.data());
            gelu_inplace(ff_hidden.data(), ff_hidden.size());

            std::vector<float> ff_out(Q.size());
            linear(ff_hidden.data(), S, intermediate_size, L.W2.data(), L.b2.data(), H, ff_out.data());
            for (size_t i = 0; i < x.size(); ++i) x[i] += ff_out[i];  // residual
            layernorm_rows(x.data(), S, H, L.ff_ln_w.data(), L.ff_ln_b.data(), layer_norm_eps);
        }

        BertOutput out;
        out.hidden = x;
        out.pooled.resize(H);
        linear(x.data(), 1, H, pooler_w.data(), pooler_b.data(), H, out.pooled.data());  // x[0]=[CLS] row
        for (float& v : out.pooled) v = std::tanh(v);
        return out;
    }
};

inline TinyBert load_bert(const std::string& artifacts_dir) {
    std::ifstream config(artifacts_dir + "/config.txt");
    if (!config) throw std::runtime_error("cannot open config.txt");
    TinyBert m;
    config >> m.num_layers >> m.hidden_size >> m.num_heads >> m.intermediate_size >> m.vocab_size >>
        m.max_position_embeddings >> m.layer_norm_eps;
    m.head_dim = m.hidden_size / m.num_heads;

    m.word_emb = load_f32(artifacts_dir + "/word_embeddings.bin",
                           static_cast<size_t>(m.vocab_size) * m.hidden_size);
    m.pos_emb = load_f32(artifacts_dir + "/position_embeddings.bin",
                          static_cast<size_t>(m.max_position_embeddings) * m.hidden_size);
    m.type_emb = load_f32(artifacts_dir + "/token_type_embeddings.bin", static_cast<size_t>(2) * m.hidden_size);
    m.emb_ln_w = load_f32(artifacts_dir + "/emb_ln_weight.bin", m.hidden_size);
    m.emb_ln_b = load_f32(artifacts_dir + "/emb_ln_bias.bin", m.hidden_size);

    for (int i = 0; i < m.num_layers; ++i) {
        std::string p = artifacts_dir + "/layer" + std::to_string(i) + "_";
        BertLayerWeights L;
        L.Wq = load_f32(p + "q_weight.bin", static_cast<size_t>(m.hidden_size) * m.hidden_size);
        L.bq = load_f32(p + "q_bias.bin", m.hidden_size);
        L.Wk = load_f32(p + "k_weight.bin", static_cast<size_t>(m.hidden_size) * m.hidden_size);
        L.bk = load_f32(p + "k_bias.bin", m.hidden_size);
        L.Wv = load_f32(p + "v_weight.bin", static_cast<size_t>(m.hidden_size) * m.hidden_size);
        L.bv = load_f32(p + "v_bias.bin", m.hidden_size);
        L.Wo = load_f32(p + "attn_out_weight.bin", static_cast<size_t>(m.hidden_size) * m.hidden_size);
        L.bo = load_f32(p + "attn_out_bias.bin", m.hidden_size);
        L.attn_ln_w = load_f32(p + "attn_ln_weight.bin", m.hidden_size);
        L.attn_ln_b = load_f32(p + "attn_ln_bias.bin", m.hidden_size);
        L.W1 = load_f32(p + "ff1_weight.bin", static_cast<size_t>(m.intermediate_size) * m.hidden_size);
        L.b1 = load_f32(p + "ff1_bias.bin", m.intermediate_size);
        L.W2 = load_f32(p + "ff2_weight.bin", static_cast<size_t>(m.hidden_size) * m.intermediate_size);
        L.b2 = load_f32(p + "ff2_bias.bin", m.hidden_size);
        L.ff_ln_w = load_f32(p + "ff_ln_weight.bin", m.hidden_size);
        L.ff_ln_b = load_f32(p + "ff_ln_bias.bin", m.hidden_size);
        m.layers.push_back(std::move(L));
    }

    m.pooler_w = load_f32(artifacts_dir + "/pooler_weight.bin", static_cast<size_t>(m.hidden_size) * m.hidden_size);
    m.pooler_b = load_f32(artifacts_dir + "/pooler_bias.bin", m.hidden_size);
    return m;
}
