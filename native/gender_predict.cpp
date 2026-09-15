// exp10: the first fully end-to-end native pipeline in this project — raw
// name string in, prediction out, zero Python anywhere in the path. Every
// prior experiment (exp1-9) fed the C++ side precomputed feature vectors;
// this uses text_features.hpp (must match python/gender_features.py's
// normalize()/bigrams()/extract_features() exactly) plus mlp_model.hpp's
// MLP struct — this "model" is just an MLP with a text-preprocessing step
// in front of it, not a new architecture, so it reuses both pieces rather
// than reimplementing the forward pass a third time.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mlp_model.hpp"
#include "text_features.hpp"

struct GenderClassifier {
    MLP model;
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> vocab_index;

    // Returns {P(male), P(female)} — matches gender_train.py's label_to_idx {"M":0,"F":1}.
    std::vector<float> predict(const std::string& name) const {
        auto features = extract_bigram_features(name, vocab_index, model.in_dim);
        return model.forward(features, 1);
    }
};

GenderClassifier load_classifier(const std::string& artifacts_dir) {
    GenderClassifier c;
    c.model = load_mlp(artifacts_dir);
    c.vocab = load_vocab(artifacts_dir + "/vocab.txt");
    if (static_cast<int>(c.vocab.size()) != c.model.in_dim) {
        throw std::runtime_error("vocab size mismatch with shapes.txt");
    }
    c.vocab_index = build_vocab_index(c.vocab);
    return c;
}

void predict_mode(const std::string& artifacts_dir, const std::string& name) {
    GenderClassifier c = load_classifier(artifacts_dir);
    auto probs = c.predict(name);
    std::string label = probs[0] > probs[1] ? "M" : "F";
    std::printf("{\"name\": \"%s\", \"pred\": \"%s\", \"prob_M\": %.6f, \"prob_F\": %.6f}\n",
                name.c_str(), label.c_str(), probs[0], probs[1]);
}

// Batch mode for exp10's equivalence check: one name per line in, writes the
// SAME native_outputs.bin format exp1-9's compare.py already reads — the
// only difference from exp9 is these predictions come from raw strings run
// through the C++ feature extractor, not precomputed Python features.
void predict_batch_mode(const std::string& artifacts_dir, const std::string& names_path) {
    GenderClassifier c = load_classifier(artifacts_dir);
    std::ifstream f(names_path);
    if (!f) throw std::runtime_error("cannot open " + names_path);

    std::vector<float> outputs;
    std::string name;
    int n = 0;
    while (std::getline(f, name)) {
        if (name.empty()) continue;
        auto probs = c.predict(name);
        outputs.insert(outputs.end(), probs.begin(), probs.end());
        ++n;
    }

    std::ofstream out(artifacts_dir + "/native_outputs.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(outputs.data()), outputs.size() * sizeof(float));
    std::cerr << "wrote native_outputs.bin (" << n << "x" << c.model.out_dim << ") from raw names\n";
}

// exp10's cold-invocation mode: load model+vocab, extract features + predict
// for ONE fixed name, exit — same contract as mlp.cpp/transformer.cpp's
// once_mode, but this one also pays feature-extraction cost, unlike exp1-9
// which only ever timed the matmul forward pass.
void once_mode(const std::string& artifacts_dir) {
    GenderClassifier c = load_classifier(artifacts_dir);
    auto probs = c.predict("sajal");
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
