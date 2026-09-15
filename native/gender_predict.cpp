// exp10: the first fully end-to-end native pipeline in this project — raw
// name string in, prediction out, zero Python anywhere in the path. Every
// prior experiment (exp1-9) fed the C++ side precomputed feature vectors;
// this ports python/gender_features.py's bigram extraction into C++ too, so
// the equivalence and cold-invocation questions can be asked about the
// REAL deployment shape (raw input), not just "given the same numbers, does
// the model agree" — matching python/gender_features.py's normalize()/
// bigrams()/extract_features() line for line, since equivalence here means
// both language's feature extractors must agree, not just the matmuls.
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.hpp"

std::string normalize(const std::string& name) {
    std::string lower;
    lower.reserve(name.size());
    for (char c : name) lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return "^" + lower + "$";
}

std::vector<std::string> bigrams(const std::string& padded) {
    std::vector<std::string> out;
    for (size_t i = 0; i + 1 < padded.size(); ++i) out.push_back(padded.substr(i, 2));
    return out;
}

std::vector<float> extract_features(const std::string& name,
                                     const std::unordered_map<std::string, int>& vocab_index,
                                     int vocab_size) {
    std::vector<float> vec(vocab_size, 0.0f);
    for (const auto& bg : bigrams(normalize(name))) {
        auto it = vocab_index.find(bg);
        if (it != vocab_index.end()) vec[it->second] += 1.0f;
    }
    return vec;
}

std::vector<std::string> load_vocab(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<std::string> vocab;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) vocab.push_back(line);
    }
    return vocab;
}

struct GenderModel {
    int in_dim, hidden_dim, out_dim;
    std::vector<float> W1, b1, W2, b2;
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> vocab_index;

    // Returns {P(male), P(female)} — matches gender_train.py's label_to_idx {"M":0,"F":1}.
    std::vector<float> predict(const std::string& name) const {
        auto features = extract_features(name, vocab_index, in_dim);
        std::vector<float> H(hidden_dim);
        linear(features.data(), 1, in_dim, W1.data(), b1.data(), hidden_dim, H.data());
        relu_inplace(H.data(), H.size());
        std::vector<float> Y(out_dim);
        linear(H.data(), 1, hidden_dim, W2.data(), b2.data(), out_dim, Y.data());
        softmax_rows(Y.data(), 1, out_dim);
        return Y;
    }
};

GenderModel load_model(const std::string& artifacts_dir) {
    std::ifstream shapes(artifacts_dir + "/shapes.txt");
    if (!shapes) throw std::runtime_error("cannot open shapes.txt");
    GenderModel m;
    int n_test;
    shapes >> m.in_dim >> m.hidden_dim >> m.out_dim >> n_test;

    m.W1 = load_f32(artifacts_dir + "/fc1_weight.bin", static_cast<size_t>(m.hidden_dim) * m.in_dim);
    m.b1 = load_f32(artifacts_dir + "/fc1_bias.bin", m.hidden_dim);
    m.W2 = load_f32(artifacts_dir + "/fc2_weight.bin", static_cast<size_t>(m.out_dim) * m.hidden_dim);
    m.b2 = load_f32(artifacts_dir + "/fc2_bias.bin", m.out_dim);

    m.vocab = load_vocab(artifacts_dir + "/vocab.txt");
    if (static_cast<int>(m.vocab.size()) != m.in_dim) {
        throw std::runtime_error("vocab size mismatch with shapes.txt");
    }
    for (size_t i = 0; i < m.vocab.size(); ++i) m.vocab_index[m.vocab[i]] = static_cast<int>(i);
    return m;
}

void predict_mode(const std::string& artifacts_dir, const std::string& name) {
    GenderModel m = load_model(artifacts_dir);
    auto probs = m.predict(name);
    std::string label = probs[0] > probs[1] ? "M" : "F";
    std::printf("{\"name\": \"%s\", \"pred\": \"%s\", \"prob_M\": %.6f, \"prob_F\": %.6f}\n",
                name.c_str(), label.c_str(), probs[0], probs[1]);
}

// Batch mode for exp10's equivalence check: one name per line in, writes the
// SAME native_outputs.bin format exp1-9's compare.py already reads — the
// only difference from exp9 is these predictions come from raw strings run
// through the C++ feature extractor, not precomputed Python features.
void predict_batch_mode(const std::string& artifacts_dir, const std::string& names_path) {
    GenderModel m = load_model(artifacts_dir);
    std::ifstream f(names_path);
    if (!f) throw std::runtime_error("cannot open " + names_path);

    std::vector<float> outputs;
    std::string name;
    int n = 0;
    while (std::getline(f, name)) {
        if (name.empty()) continue;
        auto probs = m.predict(name);
        outputs.insert(outputs.end(), probs.begin(), probs.end());
        ++n;
    }

    std::ofstream out(artifacts_dir + "/native_outputs.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(outputs.data()), outputs.size() * sizeof(float));
    std::cerr << "wrote native_outputs.bin (" << n << "x" << m.out_dim << ") from raw names\n";
}

// exp10's cold-invocation mode: load model+vocab, extract features + predict
// for ONE fixed name, exit — same contract as mlp.cpp/transformer.cpp's
// once_mode, but this one also pays feature-extraction cost, unlike exp1-9
// which only ever timed the matmul forward pass.
void once_mode(const std::string& artifacts_dir) {
    GenderModel m = load_model(artifacts_dir);
    auto probs = m.predict("sajal");
    asm volatile("" : : "g"(probs.data()) : "memory");
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage:\n"
                  << "  " << argv[0] << " <artifacts_dir> predict <name>\n"
                  << "  " << argv[0] << " <artifacts_dir> predict_batch <names_file>\n"
                  << "  " << argv[0] << " <artifacts_dir> once\n";
        return 1;
    }
    std::string artifacts_dir = argv[1], mode = argv[2];
    try {
        if (mode == "predict" && argc == 4) {
            predict_mode(artifacts_dir, argv[3]);
        } else if (mode == "predict_batch" && argc == 4) {
            predict_batch_mode(artifacts_dir, argv[3]);
        } else if (mode == "once") {
            once_mode(artifacts_dir);
        } else {
            std::cerr << "unknown mode or wrong arg count: " << mode << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
