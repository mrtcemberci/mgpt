#include "CharacterTokenizer.h"

#include <algorithm>
#include <fstream>
#include <iostream>

void CharacterTokenizer::build_vocab(const std::string& text) {
    vocab.clear();
    char_to_int.clear();
    int_to_char.clear();

    for (char c : text) {
        if (char_to_int.find(c) == char_to_int.end()) {
            vocab.push_back(c);
            char_to_int[c] = 0;
        }
    }

    std::sort(vocab.begin(), vocab.end());

    for (size_t i = 0; i < vocab.size(); ++i) {
        char_to_int[vocab[i]] = i;
        int_to_char[i] = vocab[i];
    }
}

void CharacterTokenizer::build_vocab_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file at " << filepath << "\n";
        return;
    }

    std::string text((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
    file.close();

    build_vocab(text);
}

std::vector<int> CharacterTokenizer::load_and_encode(const std::string& filepath) {
    std::vector<int> encoded_data;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file at " << filepath << "\n";
        return encoded_data;
    }

    std::string text((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
    file.close();

    return encode(text);
}

std::vector<int> CharacterTokenizer::encode(const std::string& text) {
    if (vocab.empty()) {
        build_vocab(text);
    }

    std::vector<int> encoded_data;
    encoded_data.reserve(text.size());

    for (char c : text) {
        if (char_to_int.find(c) == char_to_int.end()) {
            std::cerr << "Warning: Character '" << c << "' not recognized in vocabulary. Skipping.\n";
            continue;
        }
        encoded_data.push_back(char_to_int[c]);
    }
    return encoded_data;
}

std::string CharacterTokenizer::decode(const std::vector<int>& encoded_data) {
    std::string decoded_data;
    decoded_data.reserve(encoded_data.size());

    for (int c : encoded_data) {
        if (!int_to_char.contains(c)) {
            std::cerr << "Character ID " << c << " not recognized\n";
            continue;
        }
        decoded_data.push_back(int_to_char[c]);
    }

    return decoded_data;
}

void CharacterTokenizer::save_vocab(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) return;
    int sz = (int)vocab.size();
    out.write((const char*)&sz, sizeof(int));
    if (sz > 0) {
        out.write(vocab.data(), sz);
    }
    out.close();
}

bool CharacterTokenizer::load_vocab(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) return false;
    int sz = 0;
    in.read((char*)&sz, sizeof(int));
    if (sz <= 0) return false;
    vocab.resize(sz);
    in.read(vocab.data(), sz);
    in.close();
    char_to_int.clear();
    int_to_char.clear();
    for (int i = 0; i < sz; ++i) {
        char_to_int[vocab[i]] = i;
        int_to_char[i] = vocab[i];
    }
    return true;
}

size_t CharacterTokenizer::get_vocab_size() const {
    return vocab.size();
}

const std::vector<char>& CharacterTokenizer::get_vocab() const {
    return vocab;
}
