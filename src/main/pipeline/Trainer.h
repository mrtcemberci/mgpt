#ifndef TRAINER_H
#define TRAINER_H

#include <vector>
#include <random>
#include "CLIConfig.h"
#include "../neuralnetwork/GPT.h"
#include "../utils/Tokenizer.h"

class Trainer {
private:
    CLIConfig config;

    void get_batch(const std::vector<int>& data, int batch_size, int max_seq_len, 
                   Tensor& x_batch, Tensor& y_batch, std::mt19937& rng, const int* d_data = nullptr);

    float evaluate_loss(GPT& model, const std::vector<int>& val_data, 
                        int eval_steps, int batch_size, int max_seq_len, std::mt19937& rng, Device dev, const int* d_val_data = nullptr);

public:
    explicit Trainer(const CLIConfig& cfg);
    int run();
};

#endif // TRAINER_H
