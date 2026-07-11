#include "BytePairEncodingTokenizer.h"
#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>
#include <sstream>

BytePairEncodingTokenizer::BytePairEncodingTokenizer(int target_vocab_size)
    : target_vocab_size(target_vocab_size) {}

bool BytePairEncodingTokenizer::can_merge(const std::string& left, const std::string& right) {
    if (left.empty() || right.empty()) return false;

    // 1. Never merge across newlines (prevents speaker headers from merging into next line)
    if (left.find('\n') != std::string::npos || right.find('\n') != std::string::npos) {
        return false;
    }

    // 2. Never merge if right starts with a space (prevents merging across word boundaries like "love" + " you")
    if (right.front() == ' ') {
        return false;
    }

    // 3. If left ends with a space, only allow if left is EXACTLY " " (allows leading space on a word " " + "word" -> " word")
    if (left.back() == ' ' && left != " ") {
        return false;
    }

    auto is_alpha_num = [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
    };

    // 4. Do not merge alphanumeric characters with punctuation (keeps punctuation separate from words)
    bool left_has_alnum = false, left_has_punct = false;
    for (char c : left) {
        if (c != ' ') {
            if (is_alpha_num(c)) left_has_alnum = true;
            else left_has_punct = true;
        }
    }
    bool right_has_alnum = false, right_has_punct = false;
    for (char c : right) {
        if (c != ' ') {
            if (is_alpha_num(c)) right_has_alnum = true;
            else right_has_punct = true;
        }
    }

    if ((left_has_alnum && !left_has_punct && right_has_punct && !right_has_alnum) ||
        (left_has_punct && !left_has_alnum && right_has_alnum && !right_has_punct)) {
        return false;
    }

    // 5. Do not merge multi-character ALL-CAPS words/speaker names ("KING", "ROMEO") with lowercase words
    auto is_all_upper = [](const std::string& s) {
        int upper_count = 0;
        for (char c : s) {
            if (c >= 'A' && c <= 'Z') upper_count++;
            else if (c >= 'a' && c <= 'z') return false;
        }
        return upper_count >= 2;
    };
    auto has_lower = [](const std::string& s) {
        for (char c : s) {
            if (c >= 'a' && c <= 'z') return true;
        }
        return false;
    };

    if (is_all_upper(left) && has_lower(right)) {
        return false;
    }

    return true;
}

void BytePairEncodingTokenizer::build_vocab(const std::string& text) {
    id_to_token.clear();
    char_to_id.clear();
    vocab_chars.clear();
    ordered_merges.clear();

    std::vector<char> unique_chars;
    for (int b = 0; b < 256; ++b) {
        unique_chars.push_back((char)b);
    }

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
            if (can_merge(id_to_token[ids[i]], id_to_token[ids[i + 1]])) {
                pair_counts[{ids[i], ids[i + 1]}]++;
            }
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
    std::vector<int> final_ids;
    final_ids.reserve(text.size());

    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find('\n', start);
        if (end == std::string::npos) end = text.size();
        else end += 1; // Include newline character

        std::string line = text.substr(start, end - start);
        start = end;

        std::vector<int> ids;
        ids.reserve(line.size());
        for (char c : line) {
            auto it = char_to_id.find(c);
            if (it != char_to_id.end()) {
                ids.push_back(it->second);
            } else {
                auto space_it = char_to_id.find(' ');
                if (space_it != char_to_id.end()) {
                    ids.push_back(space_it->second);
                }
            }
        }

        for (const auto& rule : ordered_merges) {
            bool merged_any = false;
            for (size_t i = 0; i + 1 < ids.size(); ++i) {
                if (ids[i] == rule.left && ids[i + 1] == rule.right &&
                    can_merge(id_to_token[ids[i]], id_to_token[ids[i + 1]])) {
                    merged_any = true;
                    break;
                }
            }
            if (!merged_any) continue;

            std::vector<int> next_ids;
            next_ids.reserve(ids.size());
            for (size_t i = 0; i < ids.size(); ++i) {
                if (i + 1 < ids.size() && ids[i] == rule.left && ids[i + 1] == rule.right &&
                    can_merge(id_to_token[ids[i]], id_to_token[ids[i + 1]])) {
                    next_ids.push_back(rule.new_id);
                    i++;
                } else {
                    next_ids.push_back(ids[i]);
                }
            }
            ids = std::move(next_ids);
        }

        final_ids.insert(final_ids.end(), ids.begin(), ids.end());
    }

    return final_ids;
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

void BytePairEncodingTokenizer::save_vocab(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) return;
    int vocab_sz = (int)id_to_token.size();
    out.write((const char*)&vocab_sz, sizeof(int));
    for (const auto& token : id_to_token) {
        int len = (int)token.size();
        out.write((const char*)&len, sizeof(int));
        out.write(token.data(), len);
    }
    int num_merges = (int)ordered_merges.size();
    out.write((const char*)&num_merges, sizeof(int));
    for (const auto& rule : ordered_merges) {
        out.write((const char*)&rule.left, sizeof(int));
        out.write((const char*)&rule.right, sizeof(int));
        out.write((const char*)&rule.new_id, sizeof(int));
    }
    out.close();
}

bool BytePairEncodingTokenizer::load_vocab(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) return false;
    int vocab_sz = 0;
    in.read((char*)&vocab_sz, sizeof(int));
    if (vocab_sz <= 0) return false;
    id_to_token.resize(vocab_sz);
    vocab_chars.clear();
    char_to_id.clear();
    for (int i = 0; i < vocab_sz; ++i) {
        int len = 0;
        in.read((char*)&len, sizeof(int));
        std::string token(len, '\0');
        in.read(&token[0], len);
        id_to_token[i] = token;
        if (len == 1) {
            char c = token[0];
            vocab_chars.push_back(c);
            char_to_id[c] = i;
        }
    }
    int num_merges = 0;
    in.read((char*)&num_merges, sizeof(int));
    ordered_merges.resize(num_merges);
    for (int i = 0; i < num_merges; ++i) {
        in.read((char*)&ordered_merges[i].left, sizeof(int));
        in.read((char*)&ordered_merges[i].right, sizeof(int));
        in.read((char*)&ordered_merges[i].new_id, sizeof(int));
    }
    in.close();
    return true;
}

size_t BytePairEncodingTokenizer::get_vocab_size() const {
    return id_to_token.size();
}

const std::vector<char>& BytePairEncodingTokenizer::get_vocab() const {
    return vocab_chars;
}