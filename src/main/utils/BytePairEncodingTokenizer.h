#ifndef BYTEPAIRENCODINGTOKENIZER_H
#define BYTEPAIRENCODINGTOKENIZER_H

#include "Tokenizer.h"
#include <string>
#include <vector>
#include <unordered_map>

class BytePairEncodingTokenizer : public Tokenizer {
private:
    struct MergeRule {
        int left;
        int right;
        int new_id;
    };

    int target_vocab_size;
    std::vector<std::string> id_to_token;
    std::vector<char> vocab_chars; // For get_vocab() compatibility
    std::unordered_map<char, int> char_to_id;
    std::vector<MergeRule> ordered_merges;

public:
    explicit BytePairEncodingTokenizer(int target_vocab_size = 512);

    void build_vocab(const std::string& text) override;
    void build_vocab_from_file(const std::string& filepath) override;

    std::vector<int> load_and_encode(const std::string& filepath) override;
    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& encoded_data) override;

    size_t get_vocab_size() const override;
    const std::vector<char>& get_vocab() const override;
};

#endif //BYTEPAIRENCODINGTOKENIZER_H
