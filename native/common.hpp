// Shared utilities across native experiments: raw-float32 loading and the
// percentile function used by every bench_mode() (kept identical to
// python/bench_common.py's numpy.percentile default so C++ and Python
// latency numbers are directly comparable).
#pragma once

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

inline std::vector<float> load_f32(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<float> data(count);
    f.read(reinterpret_cast<char*>(data.data()), count * sizeof(float));
    if (!f) throw std::runtime_error("short read: " + path);
    return data;
}

inline double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return v[lo];
    double frac = idx - lo;
    return v[lo] * (1 - frac) + v[hi] * frac;  // matches numpy.percentile's default 'linear' method
}

// --- A tiny operator library (exactly the set the brief's §3 calls out:
// MatMul, LayerNorm, GELU, Softmax) — the whole "does a model need only a
// handful of ops" premise starts with actually having that handful shared
// and tested in one place, not reimplemented per experiment. ---

// Y[rows,out_dim] = X[rows,in_dim] @ W[out_dim,in_dim]^T + b[out_dim].
// W's [out,in] layout matches PyTorch's nn.Linear.weight storage directly,
// so CblasTrans avoids ever physically transposing a weight matrix.
inline void linear(const float* X, int rows, int in_dim, const float* W, const float* b,
                    int out_dim, float* Y) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, out_dim, in_dim, 1.0f, X, in_dim,
                W, in_dim, 0.0f, Y, out_dim);
    if (b) {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < out_dim; ++j) Y[static_cast<size_t>(i) * out_dim + j] += b[j];
    }
}

inline void relu_inplace(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i) data[i] = data[i] > 0.0f ? data[i] : 0.0f;
}

// PyTorch's default F.gelu is the exact erf formulation, not the tanh approximation.
inline void gelu_inplace(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i) data[i] = 0.5f * data[i] * (1.0f + erff(data[i] / 1.4142135f));
}

inline void softmax_rows(float* data, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float* row = data + static_cast<size_t>(i) * cols;
        float mx = *std::max_element(row, row + cols);
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            row[j] = std::exp(row[j] - mx);  // subtract row max first: numerical stability
            sum += row[j];
        }
        for (int j = 0; j < cols; ++j) row[j] /= sum;
    }
}

// exp17: elementwise add, needed for residual/skip connections (y = x +
// sublayer(x)) once the compiler's IR became a DAG instead of a flat chain.
inline void add_inplace(float* dst, const float* src, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] += src[i];
}

// Multi-head self-attention computed WITHOUT calling into cblas_sgemm.
// exp5 (research/experiments/exp5-gemm-dispatch-overhead/results.md) found
// Accelerate's per-cblas_sgemm-call dispatch cost dominates at exactly this
// matmul size (seq_len x d_head), and Accelerate has no batched-GEMM
// primitive (checked: no cblas_?gemm_batch in the SDK headers) to fold the
// per-head loop into one call. So instead of calling BLAS 2*n_heads times,
// this hand-writes the (small, memory-bound) score and context matmuls
// directly — trading "optimized kernel, called many times" for "naive
// kernel, called zero times", testing exp5's finding that at this size the
// call overhead outweighs BLAS's compute advantage.
// Q,K,V: [S,D] row-major, D = H*Dh (head h occupies columns [h*Dh,(h+1)*Dh)).
// ctx: [S,D] output. scores_scratch: reused [S,S] buffer, caller-owned.
inline void multi_head_attention_naive(const float* Q, const float* K, const float* V, float* ctx,
                                        int S, int H, int Dh, float scale, float* scores_scratch) {
    int D = H * Dh;
    for (int h = 0; h < H; ++h) {
        for (int i = 0; i < S; ++i) {
            const float* qi = Q + static_cast<size_t>(i) * D + h * Dh;
            for (int j = 0; j < S; ++j) {
                const float* kj = K + static_cast<size_t>(j) * D + h * Dh;
                float sum = 0.0f;
                for (int d = 0; d < Dh; ++d) sum += qi[d] * kj[d];
                scores_scratch[i * S + j] = sum * scale;
            }
        }
        softmax_rows(scores_scratch, S, S);
        for (int i = 0; i < S; ++i) {
            float* ctx_i = ctx + static_cast<size_t>(i) * D + h * Dh;
            for (int d = 0; d < Dh; ++d) ctx_i[d] = 0.0f;
            for (int j = 0; j < S; ++j) {
                float a = scores_scratch[i * S + j];
                const float* vj = V + static_cast<size_t>(j) * D + h * Dh;
                for (int d = 0; d < Dh; ++d) ctx_i[d] += a * vj[d];
            }
        }
    }
}

inline void layernorm_rows(float* data, int rows, int cols, const float* gamma, const float* beta,
                            float eps = 1e-5f) {
    for (int i = 0; i < rows; ++i) {
        float* row = data + static_cast<size_t>(i) * cols;
        float mean = 0.0f;
        for (int j = 0; j < cols; ++j) mean += row[j];
        mean /= cols;
        float var = 0.0f;
        for (int j = 0; j < cols; ++j) {
            float d = row[j] - mean;
            var += d * d;
        }
        var /= cols;  // biased variance (divide by cols, not cols-1) — matches PyTorch's LayerNorm
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int j = 0; j < cols; ++j) row[j] = (row[j] - mean) * inv_std * gamma[j] + beta[j];
    }
}
