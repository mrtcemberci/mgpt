#include "CrossEntropyLossLayer.h"
#include <algorithm>

void CrossEntropyLossLayer::set_targets(const std::vector<int>& targets) {
    cached_targets = targets;
    using_tensor_targets = false;
}

void CrossEntropyLossLayer::set_targets(const Tensor& targets) {
    cached_targets_tensor = targets;
    using_tensor_targets = true;
}

// FORWARD PASS: Compute numerically stable Softmax probabilities P from logits L
void CrossEntropyLossLayer::forward_into(const Tensor& input, Tensor& output) {
    if (input.shape.empty()) {
        std::cerr << "CrossEntropyLossLayer::forward_into: input tensor is empty!" << std::endl;
        exit(-1);
    }
    input.softmax_into(-1, output);
    cached_probs = output;
}

Tensor CrossEntropyLossLayer::forward(const Tensor& input) {
    forward_into(input, cached_probs);
    return cached_probs;
}

// HELPER: Compute Softmax and average Cross-Entropy loss across all tokens
float CrossEntropyLossLayer::forward_loss(const Tensor& logits, const std::vector<int>& targets) {
    set_targets(targets);
    return logits.cross_entropy_loss(cached_targets, cached_probs);
}

float CrossEntropyLossLayer::forward_loss(const Tensor& logits, const Tensor& targets) {
    set_targets(targets);
    return logits.cross_entropy_loss_into(cached_targets_tensor, cached_probs);
}

// BACKWARD PASS: Compute gradient w.r.t logits L
void CrossEntropyLossLayer::backward_into(const Tensor& dout, Tensor& din) {
    if (cached_probs.size() == 0) {
        std::cerr << "CrossEntropyLossLayer::backward_into: forward must be called before backward!" << std::endl;
        exit(-1);
    }
    if (using_tensor_targets) {
        cached_probs.cross_entropy_backward_into(cached_targets_tensor, din);
    } else {
        if (din.shape != cached_probs.shape || din.device != cached_probs.device || (!din.cuda_data && cached_probs.device == Device::CUDA)) {
            din = Tensor(cached_probs.shape, 0.0f, cached_probs.device);
        }
        din = cached_probs.cross_entropy_backward(cached_targets);
    }
    cached_dX = din;
}

Tensor CrossEntropyLossLayer::backward(const Tensor& dout) {
    backward_into(dout, cached_dX);
    return cached_dX;
}
