// The Sajal Labs CLI — Phase 6 of the roadmap (research/gap-analysis.md),
// built after exp1-10 validated the underlying equivalence/benchmark
// methodology. `sajal inspect|run|bench <artifacts_dir> [input]`.
//
// Deliberately NOT a replacement for mlp.cpp/transformer.cpp/gender_predict.cpp/
// bert.cpp — those stay exactly as they are, because exp1-14's benchmark
// numbers depend on them being minimal, dedicated, unchanged binaries (the
// whole project's finding is that generic-dispatch overhead matters; adding
// a dispatching layer to the RESEARCH binaries would contaminate the very
// thing being measured). `sajal` links the same model headers (mlp_model.hpp,
// transformer_model.hpp, text_features.hpp, bert_model.hpp, wordpiece.hpp)
// directly and does the work in-process — no shelling out to a second
// binary, which would double process-startup cost and undermine the
// cold-invocation story this project spent fourteen experiments establishing.
//
// Model "kind" is detected from what's already in the artifacts directory
// rather than a new manifest format (per research/gap-analysis.md's "don't
// invent a format before understanding why existing ones exist"): a
// config.txt (BERT's export filename, distinct from everything else's
// shapes.txt) means a WordPiece transformer (exp11-14's shape); otherwise a
// vocab.txt means bigram-MLP text input (exp9/10's shape); otherwise
// shapes.txt's field count distinguishes plain MLP (4 fields, exp1's shape)
// from a synthetic transformer block (5 fields, exp2-8's shape).
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "bert_model.hpp"
#include "mlp_model.hpp"
#include "text_features.hpp"
#include "transformer_model.hpp"
#include "wordpiece.hpp"

namespace fs = std::filesystem;

enum class Kind { TextMLP, MLP, Transformer, Bert };

Kind detect_kind(const std::string& artifacts_dir) {
    if (fs::exists(artifacts_dir + "/config.txt")) return Kind::Bert;
    if (fs::exists(artifacts_dir + "/vocab.txt")) return Kind::TextMLP;

    std::ifstream shapes(artifacts_dir + "/shapes.txt");
    if (!shapes) throw std::runtime_error("cannot open " + artifacts_dir + "/shapes.txt");
    std::string line;
    std::getline(shapes, line);
    std::istringstream iss(line);
    int n_fields = 0;
    for (std::string tok; iss >> tok; ) ++n_fields;

    if (n_fields == 4) return Kind::MLP;
    if (n_fields == 5) return Kind::Transformer;
    throw std::runtime_error("unrecognized shapes.txt format (" + std::to_string(n_fields) + " fields)");
}

const char* kind_name(Kind k) {
    switch (k) {
        case Kind::TextMLP: return "text-mlp";
        case Kind::MLP: return "mlp";
        case Kind::Transformer: return "transformer";
        case Kind::Bert: return "bert";
    }
    return "unknown";
}

// exp11-14's bench/once convention: a fixed "case1" input (real tokenized
// sentence for bert-tiny, a random-but-fixed sequence for the exp12/13
// synthetic sweep configs) sized by test_cases.txt's second line — reused
// here rather than re-tokenizing per call, matching bert.cpp's own
// BENCH_CASE=1 contract so numbers stay comparable across binaries.
std::vector<int> load_bert_bench_case(const std::string& artifacts_dir) {
    std::ifstream tc(artifacts_dir + "/test_cases.txt");
    if (!tc) throw std::runtime_error("cannot open test_cases.txt");
    std::string line;
    std::getline(tc, line);  // case0, unused
    std::getline(tc, line);  // case1, the bench case
    int seq_len = std::stoi(line.substr(0, line.find_first_of(" \t")));

    std::ifstream f(artifacts_dir + "/case1_input_ids.bin", std::ios::binary);
    if (!f) throw std::runtime_error("cannot open case1_input_ids.bin");
    std::vector<int32_t> raw(seq_len);
    f.read(reinterpret_cast<char*>(raw.data()), seq_len * sizeof(int32_t));
    return std::vector<int>(raw.begin(), raw.end());
}

bool has_tokenizer(const std::string& artifacts_dir) {
    return fs::exists(artifacts_dir + "/vocab.txt") && fs::exists(artifacts_dir + "/tokenizer_config.txt");
}

void inspect_cmd(const std::string& artifacts_dir) {
    Kind kind = detect_kind(artifacts_dir);
    std::printf("kind: %s\n", kind_name(kind));

    uintmax_t total_bytes = 0;
    for (auto& entry : fs::directory_iterator(artifacts_dir)) {
        if (entry.is_regular_file()) total_bytes += entry.file_size();
    }
    std::printf("artifact size: %.3f MB\n", total_bytes / 1e6);

    if (kind == Kind::TextMLP) {
        MLP m = load_mlp(artifacts_dir);
        auto vocab = load_vocab(artifacts_dir + "/vocab.txt");
        std::printf("in_dim (vocab size): %d\nhidden_dim: %d\nout_dim: %d\n", m.in_dim, m.hidden_dim, m.out_dim);
        std::printf("vocab entries: %zu\n", vocab.size());
        std::printf("run with: sajal run %s <text>\n", artifacts_dir.c_str());
    } else if (kind == Kind::MLP) {
        MLP m = load_mlp(artifacts_dir);
        std::printf("in_dim: %d\nhidden_dim: %d\nout_dim: %d\n", m.in_dim, m.hidden_dim, m.out_dim);
        std::printf("no text/raw-input pipeline defined for this artifact — run() uses a placeholder input\n");
    } else if (kind == Kind::Transformer) {
        TransformerBlock m = load_transformer(artifacts_dir);
        std::printf("seq_len: %d\nd_model: %d\nn_heads: %d\nd_ff: %d\n", m.seq_len, m.d_model, m.n_heads, m.d_ff);
        std::printf("no text/raw-input pipeline defined for this artifact — run() uses a placeholder input\n");
    } else {
        TinyBert m = load_bert(artifacts_dir);
        long params = static_cast<long>(m.word_emb.size() + m.pos_emb.size() + m.type_emb.size());
        for (auto& L : m.layers) {
            params += L.Wq.size() + L.bq.size() + L.Wk.size() + L.bk.size() + L.Wv.size() + L.bv.size() +
                      L.Wo.size() + L.bo.size() + L.W1.size() + L.b1.size() + L.W2.size() + L.b2.size();
        }
        std::printf("num_layers: %d\nhidden_size: %d\nnum_heads: %d\nintermediate_size: %d\nvocab_size: %d\n",
                    m.num_layers, m.hidden_size, m.num_heads, m.intermediate_size, m.vocab_size);
        std::printf("params: ~%ld\n", params);
        if (has_tokenizer(artifacts_dir)) {
            std::printf("tokenizer: available — run with: sajal run %s <text>\n", artifacts_dir.c_str());
        } else {
            std::printf("tokenizer: not available (synthetic benchmark artifact) — bench only, no run()\n");
        }
    }
}

void run_cmd(const std::string& artifacts_dir, const std::vector<std::string>& extra_args) {
    Kind kind = detect_kind(artifacts_dir);
    if (kind == Kind::TextMLP) {
        if (extra_args.empty()) throw std::runtime_error("this model takes text input: sajal run <dir> <text>");
        MLP m = load_mlp(artifacts_dir);
        auto vocab = load_vocab(artifacts_dir + "/vocab.txt");
        auto vocab_index = build_vocab_index(vocab);
        auto features = extract_bigram_features(extra_args[0], vocab_index, m.in_dim);
        auto probs = m.forward(features, 1);
        std::printf("{\"input\": \"%s\", \"output\": [", extra_args[0].c_str());
        for (size_t i = 0; i < probs.size(); ++i) std::printf("%s%.6f", i ? ", " : "", probs[i]);
        std::printf("]}\n");
    } else if (kind == Kind::MLP) {
        MLP m = load_mlp(artifacts_dir);
        std::vector<float> x(m.in_dim, 0.1f);
        auto y = m.forward(x, 1);
        std::printf("{\"note\": \"no text pipeline for this artifact, using placeholder input\", \"output\": [");
        for (size_t i = 0; i < y.size(); ++i) std::printf("%s%.6f", i ? ", " : "", y[i]);
        std::printf("]}\n");
    } else if (kind == Kind::Transformer) {
        TransformerBlock m = load_transformer(artifacts_dir);
        std::vector<float> x(static_cast<size_t>(m.seq_len) * m.d_model, 0.1f);
        const auto& y = m.forward(x);
        float mn = y[0], mx = y[0], sum = 0;
        for (float v : y) { mn = std::min(mn, v); mx = std::max(mx, v); sum += v; }
        std::printf("{\"note\": \"no text pipeline for this artifact, using placeholder input\", "
                    "\"output_shape\": [%d, %d], \"output_min\": %.6f, \"output_max\": %.6f, \"output_mean\": %.6f}\n",
                    m.seq_len, m.d_model, mn, mx, sum / y.size());
    } else {
        if (!has_tokenizer(artifacts_dir)) {
            throw std::runtime_error("no tokenizer (vocab.txt/tokenizer_config.txt) in this artifact — "
                                      "this looks like a synthetic benchmark-only export, not a real model");
        }
        if (extra_args.empty()) throw std::runtime_error("this model takes text input: sajal run <dir> <text>");
        WordPieceTokenizer tok = load_tokenizer(artifacts_dir);
        TinyBert m = load_bert(artifacts_dir);
        auto ids = tok.tokenize(extra_args[0]);
        auto out = m.forward(ids);
        std::printf("{\"input\": \"%s\", \"n_tokens\": %zu, \"pooled_preview\": [", extra_args[0].c_str(), ids.size());
        for (int i = 0; i < 5 && i < static_cast<int>(out.pooled.size()); ++i)
            std::printf("%s%.6f", i ? ", " : "", out.pooled[i]);
        std::printf("]}\n");
    }
}

// Same warmup+timed-loop contract as mlp.cpp/transformer.cpp/bert.cpp's
// bench_mode, applied generically via detect_kind() so any artifact can be
// benchmarked through one command instead of remembering which binary each
// kind needs.
void bench_cmd(const std::string& artifacts_dir) {
    Kind kind = detect_kind(artifacts_dir);
    const int WARMUP = 50, ITERS = 500;
    auto t_start = Clock::now();

    std::vector<double> latencies_ms;
    latencies_ms.reserve(ITERS);
    double cold_start_ms;

    auto run_loop = [&](auto&& forward_once) {
        cold_start_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
        for (int i = 0; i < WARMUP; ++i) forward_once();
        for (int i = 0; i < ITERS; ++i) {
            auto t0 = Clock::now();
            forward_once();
            latencies_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        }
    };

    if (kind == Kind::TextMLP) {
        MLP m = load_mlp(artifacts_dir);
        auto vocab = load_vocab(artifacts_dir + "/vocab.txt");
        auto vocab_index = build_vocab_index(vocab);
        run_loop([&] {
            auto features = extract_bigram_features("sajal", vocab_index, m.in_dim);
            auto y = m.forward(features, 1);
            asm volatile("" : : "g"(y.data()) : "memory");
        });
    } else if (kind == Kind::MLP) {
        MLP m = load_mlp(artifacts_dir);
        std::vector<float> x(m.in_dim, 0.1f);
        run_loop([&] {
            auto y = m.forward(x, 1);
            asm volatile("" : : "g"(y.data()) : "memory");
        });
    } else if (kind == Kind::Transformer) {
        TransformerBlock m = load_transformer(artifacts_dir);
        std::vector<float> x(static_cast<size_t>(m.seq_len) * m.d_model, 0.1f);
        run_loop([&] {
            const auto& y = m.forward(x);
            asm volatile("" : : "g"(y.data()) : "memory");
        });
    } else {
        TinyBert m = load_bert(artifacts_dir);
        auto token_ids = load_bert_bench_case(artifacts_dir);
        run_loop([&] {
            auto out = m.forward(token_ids);
            asm volatile("" : : "g"(out.pooled.data()) : "memory");
        });
    }

    double mean = 0;
    for (double v : latencies_ms) mean += v;
    mean /= latencies_ms.size();
    std::printf(
        "{\"kind\": \"%s\", \"cold_start_ms\": %.4f, \"warmup_iters\": %d, \"iters\": %d, "
        "\"mean_ms\": %.5f, \"p50_ms\": %.5f, \"p90_ms\": %.5f, \"p95_ms\": %.5f, \"p99_ms\": %.5f}\n",
        kind_name(kind), cold_start_ms, WARMUP, ITERS, mean, percentile(latencies_ms, 50),
        percentile(latencies_ms, 90), percentile(latencies_ms, 95), percentile(latencies_ms, 99));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage:\n"
                  << "  " << argv[0] << " inspect <artifacts_dir>\n"
                  << "  " << argv[0] << " run <artifacts_dir> [text]\n"
                  << "  " << argv[0] << " bench <artifacts_dir>\n";
        return 1;
    }
    std::string cmd = argv[1], artifacts_dir = argv[2];
    std::vector<std::string> extra(argv + 3, argv + argc);
    try {
        if (cmd == "inspect") {
            inspect_cmd(artifacts_dir);
        } else if (cmd == "run") {
            run_cmd(artifacts_dir, extra);
        } else if (cmd == "bench") {
            bench_cmd(artifacts_dir);
        } else {
            std::cerr << "unknown command: " << cmd << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
