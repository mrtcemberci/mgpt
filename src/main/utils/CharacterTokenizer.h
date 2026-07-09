#ifndef CHARACTERTOKENIZER_H
#define CHARACTERTOKENIZER_H

#include "Tokenizer.h"
#include <string>
#include <unordered_map>
#include <vector>

class CharacterTokenizer : public Tokenizer {
private:
    std::vector<char> vocab;
    std::unordered_map<char, int> char_to_int;
    std::unordered_map<int, char> int_to_char;

public:
    void build_vocab(const std::string& text) override;
    void build_vocab_from_file(const std::string& filepath) override;

    std::vector<int> load_and_encode(const std::string& filepath) override;
    std::vector<int> encode(const std::string& text) override;
    std::string decode(const std::vector<int>& encoded_data) override;

    size_t get_vocab_size() const override;
    const std::vector<char>& get_vocab() const;
};

#endif //CHARACTERTOKENIZER_H
