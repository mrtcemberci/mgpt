#include "DataLoader.h"

#include <algorithm>
#include <fstream>
#include <iostream>

std::vector<int> DataLoader::load_and_encode(const std::string& filepath) {
    std::vector<int> encoded_data;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file at " << filepath << "\n";
        return encoded_data;
    }

    // reads file into the text string
    std::string text((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
    file.close();

    return encode(text);
}

std::vector<int> DataLoader::encode(const std::string& text) {
    std::vector<int> encoded_data;

    for (char c : text) {
        if (char_to_int.find(c) == char_to_int.end()) {
            vocab.push_back(c);
            char_to_int[c] = 0;
        }
    }

    // we now have a vocab vector of all unique characters, with placeholders for them in the map
    // we now need to assign IDs

    std::sort(vocab.begin(), vocab.end());

    for (size_t i = 0; i < vocab.size(); ++i) {
        char_to_int[vocab[i]] = i;
        int_to_char[i] = vocab[i];
    }

    encoded_data.reserve(text.size());

    for (char c : text) {
        encoded_data.push_back(char_to_int[c]);
    }
    return encoded_data;
}

std::string DataLoader::decode(const std::vector<int>& encoded_data) {
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
