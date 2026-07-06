#ifndef CROSSENTROPYLOSSLAYER_H
#define CROSSENTROPYLOSSLAYER_H

#include "Layer.h"
#include <vector>
#include <cmath>
#include <iostream>

class CrossEntropyLossLayer : public Layer {
private:
    Tensor cached_probs;               // Softmax probabilities P of shape {B, T, V}
    std::vector<int> cached_targets;   // Target integer class IDs Y of size B * T

public:
    CrossEntropyLossLayer() = default;

    // Set targets from std::vector<int> or from a Tensor containing float-cast integer IDs
    void set_targets(const std::vector<int>& targets);
    void set_targets(const Tensor& targets);

    // computes numerically stable Softmax probabilities P from logits L
    Tensor forward(const Tensor& input) override;

    // compute Softmax probabilities and evaluate average Cross-Entropy loss in one call
    float forward_loss(const Tensor& logits, const std::vector<int>& targets);
    float forward_loss(const Tensor& logits, const Tensor& targets);

    // computes dL = (P - 1) / (B * T) at target indices
    Tensor backward(const Tensor& dout) override;

    // No learnable weights or biases in CrossEntropyLossLayer
    std::vector<Tensor*> get_parameters() override { return {}; }
};

#endif //CROSSENTROPYLOSSLAYER_H
