#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>

// Needs a function that takes in a string file path, returns a vector of encoded integer of file contents
// Needs a function that takes in a vector int and returns a decoded string
class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual void build_vocab(const std::string& text) = 0;
    virtual void build_vocab_from_file(const std::string& filepath) = 0;

    virtual std::vector<int> load_and_encode(const std::string& filepath) = 0;
    virtual std::vector<int> encode(const std::string& text) = 0;
    virtual std::string decode(const std::vector<int>& encoded_data) = 0;

    // Getters for neural network / bigram model initialization
    virtual size_t get_vocab_size() const = 0;
    virtual const std::vector<char>& get_vocab() const = 0;
};

#endif //TOKENIZER_H
