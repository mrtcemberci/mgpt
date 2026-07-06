#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <unordered_map>
#include <vector>

// Needs a function that takes in a string file path, returns a vector of encoded integer of file contents
// Needs a function that takes in a vector int and returns a decoded string
class Tokenizer {
private:
    std::vector<char> vocab;

    std::unordered_map<char, int> char_to_int;
    std::unordered_map<int, char> int_to_char;
public:
    void build_vocab(const std::string& text);
    void build_vocab_from_file(const std::string& filepath);

    std::vector<int> load_and_encode(const std::string& filepath);
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& encoded_data);

    // Getters for neural network / bigram model initialization
    size_t get_vocab_size() const;
    const std::vector<char>& get_vocab() const;
};

#endif //TOKENIZER_H
