#include "CrossEntropyLossLayer.h"
#include <algorithm>

void CrossEntropyLossLayer::set_targets(const std::vector<int>& targets) {
    cached_targets = targets;
}

void CrossEntropyLossLayer::set_targets(const Tensor& targets) {
    cached_targets.resize(targets.size());
    for (size_t i = 0; i < targets.size(); ++i) {
        cached_targets[i] = (int)targets.data[i];
    }
}

// FORWARD PASS: Compute numerically stable Softmax probabilities P from logits L
Tensor CrossEntropyLossLayer::forward(const Tensor& input) {
    if (input.shape.empty()) {
        std::cerr << "CrossEntropyLossLayer::forward: input tensor is empty!" << std::endl;
        exit(-1);
    }
    cached_probs = input.softmax(-1);
    return cached_probs;
}

// HELPER: Compute Softmax and average Cross-Entropy loss across all tokens
float CrossEntropyLossLayer::forward_loss(const Tensor& logits, const std::vector<int>& targets) {
    set_targets(targets);
    return logits.cross_entropy_loss(cached_targets, cached_probs);
}

float CrossEntropyLossLayer::forward_loss(const Tensor& logits, const Tensor& targets) {
    set_targets(targets);
    return forward_loss(logits, cached_targets);
}

// BACKWARD PASS: Compute gradient w.r.t logits L
Tensor CrossEntropyLossLayer::backward(const Tensor& dout) {
    if (cached_probs.data.empty()) {
        std::cerr << "CrossEntropyLossLayer::backward: forward must be called before backward!" << std::endl;
        exit(-1);
    }
    Tensor dL = cached_probs.cross_entropy_backward(cached_targets);
    if (!dout.shape.empty() && dout.size() == 1 && dout.data[0] != 1.0f) {
        dL = dL * dout.data[0];
    }
    return dL;
}
