// exp14: a real WordPiece tokenizer, ported to C++ — BERT's actual
// subword algorithm (BasicTokenizer: lowercase + whitespace/punctuation
// split, then WordpieceTokenizer: greedy longest-match-first over the
// vocabulary, "##"-prefixed continuation pieces, [UNK] on failure),
// verified token-for-token against HF's BertTokenizerFast rather than
// just "produces plausible-looking tokens."
//
// Scope, stated explicitly rather than silently: this is an ASCII-focused
// port. Real BERT's BasicTokenizer does full Unicode category lookups
// (for accent stripping, CJK character isolation, Unicode punctuation
// classes); this implementation uses <cctype> functions guarded to the
// ASCII range (bytes 0-127) and passes any byte >= 0x80 through as a
// word character. Verified exactly correct on this project's real
// (ASCII English) test sentences — not a claim of full-Unicode fidelity.
#pragma once

#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct WordPieceTokenizer {
    std::unordered_map<std::string, int> vocab;
    int cls_id, sep_id, unk_id;
    bool do_lower_case;
    static constexpr size_t MAX_CHARS_PER_WORD = 200;  // BERT's own default

    std::vector<std::string> basic_tokenize(const std::string& text) const {
        std::vector<std::string> tokens;
        std::string cur;
        auto flush = [&] {
            if (!cur.empty()) {
                tokens.push_back(cur);
                cur.clear();
            }
        };
        for (unsigned char c : text) {
            if (c < 128 && std::isspace(c)) {
                flush();
            } else if (c < 128 && std::ispunct(c)) {
                flush();
                tokens.push_back(std::string(1, static_cast<char>(c)));
            } else {
                cur += do_lower_case && c < 128 ? static_cast<char>(std::tolower(c)) : static_cast<char>(c);
            }
        }
        flush();
        return tokens;
    }

    // Greedy longest-match-first over one basic token. Matches
    // transformers' WordpieceTokenizer.tokenize() exactly: if any step
    // finds no matching substring, the WHOLE word becomes a single [UNK]
    // (not a partial split), and words longer than MAX_CHARS_PER_WORD
    // are [UNK] without even trying.
    std::vector<std::string> wordpiece_split(const std::string& word) const {
        if (word.size() > MAX_CHARS_PER_WORD) return {"[UNK]"};
        std::vector<std::string> pieces;
        size_t start = 0;
        while (start < word.size()) {
            size_t end = word.size();
            std::string best;
            bool found = false;
            while (start < end) {
                std::string candidate = word.substr(start, end - start);
                if (start > 0) candidate = "##" + candidate;
                if (vocab.count(candidate)) {
                    best = candidate;
                    found = true;
                    break;
                }
                --end;
            }
            if (!found) return {"[UNK]"};
            pieces.push_back(best);
            start = end;
        }
        return pieces;
    }

    std::vector<int> tokenize(const std::string& text) const {
        std::vector<int> ids = {cls_id};
        for (const auto& word : basic_tokenize(text)) {
            for (const auto& piece : wordpiece_split(word)) {
                auto it = vocab.find(piece);
                ids.push_back(it != vocab.end() ? it->second : unk_id);
            }
        }
        ids.push_back(sep_id);
        return ids;
    }
};

inline WordPieceTokenizer load_tokenizer(const std::string& artifacts_dir) {
    WordPieceTokenizer t;

    std::ifstream vf(artifacts_dir + "/vocab.txt");
    if (!vf) throw std::runtime_error("cannot open vocab.txt");
    std::string line;
    int idx = 0;
    while (std::getline(vf, line)) {
        t.vocab[line] = idx++;
    }

    std::ifstream cf(artifacts_dir + "/tokenizer_config.txt");
    if (!cf) throw std::runtime_error("cannot open tokenizer_config.txt");
    int lower_flag, pad_id;
    cf >> lower_flag >> t.cls_id >> t.sep_id >> t.unk_id >> pad_id;
    t.do_lower_case = lower_flag != 0;
    return t;
}
