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
