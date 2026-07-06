#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <vector>
#include "utils/Tokenizer.h"

// Helper function to create temporary test files
void create_temp_file(const std::string& filepath, const std::string& content) {
    std::ofstream out(filepath);
    out << content;
    out.close();
}

// Test 1: Testing loading and decoding an empty file
void test_empty_file() {
    std::string filepath = "temp_empty_test.txt";
    create_temp_file(filepath, "");

    Tokenizer tokenizer;
    std::vector<int> encoded = tokenizer.load_and_encode(filepath);
    assert(encoded.empty() && "Encoded vector should be empty for an empty file");
    
    std::string decoded = tokenizer.decode(encoded);
    assert(decoded.empty() && "Decoded string should be empty");

    std::remove(filepath.c_str());
    std::cout << "Test 1 (Empty File) passed! ✅\n";
}

// Test 2: Testing a single character directly using encode(text)
void test_single_char() {
    Tokenizer tokenizer;
    std::vector<int> encoded = tokenizer.encode("a");
    assert(encoded.size() == 1 && "Encoded vector should have exactly 1 element");
    assert(encoded[0] == 0 && "First character in sorted vocab should get ID 0");

    std::string decoded = tokenizer.decode(encoded);
    assert(decoded == "a" && "Decoded string should match original 'a'");

    std::cout << "Test 2 (Single Character - Direct String Encode) passed! ✅\n";
}

// Test 3: Testing repeated identical characters directly using encode(text)
void test_repeated_chars() {
    Tokenizer tokenizer;
    std::vector<int> encoded = tokenizer.encode("zzz");
    assert(encoded.size() == 3 && "Encoded vector should have 3 elements");
    assert(encoded[0] == 0 && encoded[1] == 0 && encoded[2] == 0 && "All identical characters should get identical ID 0");

    std::string decoded = tokenizer.decode(encoded);
    assert(decoded == "zzz" && "Decoded string should match original 'zzz'");

    std::cout << "Test 3 (Repeated Characters - Direct String Encode) passed! ✅\n";
}

// Test 4: Testing vocabulary alphabetical sorting directly using encode(text)
void test_multiple_chars_sorted_vocab() {
    Tokenizer tokenizer;
    std::vector<int> encoded = tokenizer.encode("cba");
    assert(encoded.size() == 3 && "Encoded vector should have 3 elements");
    // Vocab sorted alphabetically: 'a'=0, 'b'=1, 'c'=2. So "cba" -> {2, 1, 0}
    assert(encoded[0] == 2 && "Char 'c' should get ID 2");
    assert(encoded[1] == 1 && "Char 'b' should get ID 1");
    assert(encoded[2] == 0 && "Char 'a' should get ID 0");

    std::string decoded = tokenizer.decode(encoded);
    assert(decoded == "cba" && "Decoded string should match original 'cba'");

    std::cout << "Test 4 (Sorted Vocab - Direct String Encode) passed! ✅\n";
}

// Test 5: Testing error handling when a file does not exist
void test_nonexistent_file() {
    Tokenizer tokenizer;
    std::vector<int> encoded = tokenizer.load_and_encode("this_file_does_not_exist_9999.txt");
    assert(encoded.empty() && "Should gracefully return empty vector for non-existent file");

    std::cout << "Test 5 (Non-existent File) passed! ✅\n";
}

// Test 6: Testing get_vocab_size() for neural network / bigram model initialization
void test_vocab_size_getter() {
    Tokenizer tokenizer;
    tokenizer.encode("hello world!"); // 9 unique characters: ' ', '!', 'd', 'e', 'h', 'l', 'o', 'r', 'w'
    assert(tokenizer.get_vocab_size() == 9 && "Vocab size should equal exact number of unique characters (9)");
    assert(tokenizer.get_vocab().size() == 9 && "get_vocab vector size should also be 9");

    std::cout << "Test 6 (Vocab Size Getter for Bigram Model) passed! ✅\n";
}

// Test 7: Testing separate vocabulary building and frozen encoding
void test_separate_build_vocab_and_encode() {
    Tokenizer tokenizer;
    tokenizer.build_vocab("cba"); // Vocab is frozen: 'a'=0, 'b'=1, 'c'=2
    assert(tokenizer.get_vocab_size() == 3 && "Vocab size should be 3");

    // Encoding a string with known characters
    std::vector<int> encoded = tokenizer.encode("ab");
    assert(encoded.size() == 2 && encoded[0] == 0 && encoded[1] == 1 && "Encoding should use existing vocab");

    // Encoding a string with an unknown character ('d' is not in vocab) should warn and skip 'd' without mutating vocab!
    std::vector<int> encoded_with_unk = tokenizer.encode("abd");
    assert(encoded_with_unk.size() == 2 && encoded_with_unk[0] == 0 && encoded_with_unk[1] == 1 && "Should skip unknown char 'd'");
    assert(tokenizer.get_vocab_size() == 3 && "Vocab size must remain frozen at 3!");

    std::cout << "Test 7 (Separate Build Vocab & Frozen Encoding) passed! ✅\n";
}

int main() {
    std::cout << "--- Starting Tokenizer Tests ---\n";
    test_empty_file();
    test_single_char();
    test_repeated_chars();
    test_multiple_chars_sorted_vocab();
    test_nonexistent_file();
    test_vocab_size_getter();
    test_separate_build_vocab_and_encode();
    std::cout << "--- All 7 Tests Passed Successfully! --- 🎉\n";
    return 0;
}
