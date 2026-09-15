// Character-bigram bag-of-ngrams text featurizer, factored out of
// gender_predict.cpp — general-purpose despite its origin (any MLP text
// classifier trained with python/gender_features.py's identical scheme can
// reuse this), not specific to gender classification. Must match
// python/gender_features.py's normalize()/bigrams()/extract_features()
// exactly — see exp10 (research/experiments/exp10-e2e-native-pipeline) for
// why this line-for-line correspondence matters and how it was verified.
#pragma once

#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

inline std::string text_normalize(const std::string& s) {
    std::string lower;
    lower.reserve(s.size());
    for (char c : s) lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return "^" + lower + "$";
}

inline std::vector<std::string> text_bigrams(const std::string& padded) {
    std::vector<std::string> out;
    for (size_t i = 0; i + 1 < padded.size(); ++i) out.push_back(padded.substr(i, 2));
    return out;
}

inline std::vector<std::string> load_vocab(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<std::string> vocab;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) vocab.push_back(line);
    }
    return vocab;
}

inline std::unordered_map<std::string, int> build_vocab_index(const std::vector<std::string>& vocab) {
    std::unordered_map<std::string, int> index;
    for (size_t i = 0; i < vocab.size(); ++i) index[vocab[i]] = static_cast<int>(i);
    return index;
}

inline std::vector<float> extract_bigram_features(const std::string& text,
                                                    const std::unordered_map<std::string, int>& vocab_index,
                                                    int vocab_size) {
    std::vector<float> vec(vocab_size, 0.0f);
    for (const auto& bg : text_bigrams(text_normalize(text))) {
        auto it = vocab_index.find(bg);
        if (it != vocab_index.end()) vec[it->second] += 1.0f;
    }
    return vec;
}
