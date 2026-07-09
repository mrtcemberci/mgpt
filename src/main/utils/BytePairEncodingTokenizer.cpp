#include "BytePairEncodingTokenizer.h"
#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>
#include <sstream>

BytePairEncodingTokenizer::BytePairEncodingTokenizer(int target_vocab_size)
    : target_vocab_size(target_vocab_size) {}

void BytePairEncodingTokenizer::build_vocab(const std::string& text) {
    id_to_token.clear();
    char_to_id.clear();
    ordered_merges.clear();
    vocab_chars.clear();

    // Build initial single-character base vocabulary
    std::vector<char> unique_chars;
    for (char c : text) {
        if (char_to_id.find(c) == char_to_id.end()) {
            unique_chars.push_back(c);
            char_to_id[c] = 0; // Temporary mark
        }
    }
    std::sort(unique_chars.begin(), unique_chars.end());

    for (char c : unique_chars) {
        char_to_id[c] = (int)id_to_token.size();
        id_to_token.push_back(std::string(1, c));
        vocab_chars.push_back(c);
    }

    // Convert text to initial ID sequence
    std::vector<int> ids;
    ids.reserve(text.size());
    for (char c : text) {
        ids.push_back(char_to_id[c]);
    }

    // Iterative BPE merging loop
    while ((int)id_to_token.size() < target_vocab_size) {
        std::map<std::pair<int, int>, int> pair_counts;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            pair_counts[{ids[i], ids[i + 1]}]++;
        }

        std::pair<int, int> best_pair = {-1, -1};
        int max_count = -1;
        for (const auto& [pair, count] : pair_counts) {
            if (count > max_count) {
                max_count = count;
                best_pair = pair;
            }
        }

        if (max_count < 2) {
            break; // No repeated adjacent pairs remaining
        }

        int new_id = (int)id_to_token.size();
        std::string merged_str = id_to_token[best_pair.first] + id_to_token[best_pair.second];
        id_to_token.push_back(merged_str);
        ordered_merges.push_back({best_pair.first, best_pair.second, new_id});

        // Apply merge to the current ID sequence
        std::vector<int> next_ids;
        next_ids.reserve(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i + 1 < ids.size() && ids[i] == best_pair.first && ids[i + 1] == best_pair.second) {
                next_ids.push_back(new_id);
                i++;
            } else {
                next_ids.push_back(ids[i]);
            }
        }
        ids = std::move(next_ids);
    }
}

void BytePairEncodingTokenizer::build_vocab_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file at " << filepath << "\n";
        return;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    build_vocab(buffer.str());
}

std::vector<int> BytePairEncodingTokenizer::load_and_encode(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file at " << filepath << "\n";
        return {};
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    std::string content = buffer.str();
    if (id_to_token.empty()) {
        build_vocab(content);
    }

    return encode(content);
}

std::vector<int> BytePairEncodingTokenizer::encode(const std::string& text) {
    std::vector<int> ids;
    ids.reserve(text.size());
    for (char c : text) {
        auto it = char_to_id.find(c);
        if (it != char_to_id.end()) {
            ids.push_back(it->second);
        } else {
            std::cerr << "Warning: Character '" << c << "' not found in BPE vocabulary. Skipping.\n";
        }
    }

    for (const auto& rule : ordered_merges) {
        std::vector<int> next_ids;
        next_ids.reserve(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i + 1 < ids.size() && ids[i] == rule.left && ids[i + 1] == rule.right) {
                next_ids.push_back(rule.new_id);
                i++;
            } else {
                next_ids.push_back(ids[i]);
            }
        }
        ids = std::move(next_ids);
    }

    return ids;
}

std::string BytePairEncodingTokenizer::decode(const std::vector<int>& encoded_data) {
    std::string result;
    for (int id : encoded_data) {
        if (id >= 0 && id < (int)id_to_token.size()) {
            result += id_to_token[id];
        } else {
            std::cerr << "Token ID " << id << " not recognized\n";
            continue;
        }
    }
    return result;
}

size_t BytePairEncodingTokenizer::get_vocab_size() const {
    return id_to_token.size();
}

const std::vector<char>& BytePairEncodingTokenizer::get_vocab() const {
    return vocab_chars;
}