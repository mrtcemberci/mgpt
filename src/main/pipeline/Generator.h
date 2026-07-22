#ifndef GENERATOR_H
#define GENERATOR_H

#include <string>
#include <random>
#include "CLIConfig.h"
#include "../neuralnetwork/GPT.h"
#include "../utils/Tokenizer.h"

class Generator {
private:
    CLIConfig config;

    std::string generate_text(GPT& model, Tokenizer& tokenizer, const std::string& prompt, 
                              int max_new_tokens, float temperature, int top_k, std::mt19937& rng);

public:
    explicit Generator(const CLIConfig& cfg);
    int run();
};

#endif // GENERATOR_H
