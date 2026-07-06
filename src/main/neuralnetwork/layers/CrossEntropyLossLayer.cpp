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
// P_{i, v} = exp(L_{i, v} - max_k L_{i, k}) / sum_j exp(L_{i, j} - max_k L_{i, k})
Tensor CrossEntropyLossLayer::forward(const Tensor& input) {
    if (input.shape.empty()) {
        std::cerr << "CrossEntropyLossLayer::forward: input tensor is empty!" << std::endl;
        exit(-1);
    }

    int V = input.shape.back(); // Vocabulary size / number of classes
    int total_tokens = (int)(input.size() / V);

    cached_probs = Tensor(input.shape, 0.0f);

    for (int i = 0; i < total_tokens; ++i) {
        const float* logit_ptr = input.data.data() + i * V;
        float* prob_ptr = cached_probs.data.data() + i * V;

        // Find max logit for numerical stability to prevent exp overflow
        float max_logit = logit_ptr[0];
        for (int v = 1; v < V; ++v) {
            if (logit_ptr[v] > max_logit) {
                max_logit = logit_ptr[v];
            }
        }

        // Compute exponentials and sum of exponentials
        float sum_exp = 0.0f;
        for (int v = 0; v < V; ++v) {
            float e = std::exp(logit_ptr[v] - max_logit);
            prob_ptr[v] = e;
            sum_exp += e;
        }

        // Normalize to get valid probability distribution
        float inv_sum = 1.0f / sum_exp;
        for (int v = 0; v < V; ++v) {
            prob_ptr[v] *= inv_sum;
        }
    }

    return cached_probs;
}

// HELPER: Compute Softmax and average Cross-Entropy loss across all tokens
// L = -1/(B*T) sum_{b, t} log(P_{b, t, Y_{b, t}})
float CrossEntropyLossLayer::forward_loss(const Tensor& logits, const std::vector<int>& targets) {
    set_targets(targets);
    forward(logits);

    int V = logits.shape.back();
    int total_tokens = (int)(logits.size() / V);

    if ((int)cached_targets.size() != total_tokens) {
        std::cerr << "CrossEntropyLossLayer::forward_loss: target size mismatch!" << std::endl;
        exit(-1);
    }

    float total_loss = 0.0f;
    for (int i = 0; i < total_tokens; ++i) {
        int target_class = cached_targets[i];
        if (target_class < 0 || target_class >= V) {
            std::cerr << "CrossEntropyLossLayer::forward_loss: target class out of bounds: " << target_class << std::endl;
            exit(-1);
        }
        float p = cached_probs.data[i * V + target_class];
        total_loss -= std::log(p + 1e-15f); // Add small epsilon to prevent log(0)
    }

    return total_loss / (float)total_tokens;
}

float CrossEntropyLossLayer::forward_loss(const Tensor& logits, const Tensor& targets) {
    set_targets(targets);
    return forward_loss(logits, cached_targets);
}

// BACKWARD PASS: Compute gradient w.r.t logits L
// dL_{i, v} = 1/(B*T) * (P_{i, v} - 1) if v == Y_i else 1/(B*T) * P_{i, v}
Tensor CrossEntropyLossLayer::backward(const Tensor& dout) {
    if (cached_probs.data.empty()) {
        std::cerr << "CrossEntropyLossLayer::backward: forward must be called before backward!" << std::endl;
        exit(-1);
    }

    int V = cached_probs.shape.back();
    int total_tokens = (int)(cached_probs.size() / V);

    if ((int)cached_targets.size() != total_tokens) {
        std::cerr << "CrossEntropyLossLayer::backward: target size mismatch!" << std::endl;
        exit(-1);
    }

    Tensor dL = Tensor(cached_probs.shape, 0.0f);
    float scale = 1.0f / (float)total_tokens;

    // Support optional scaling if an upstream scalar gradient dout is passed in
    if (!dout.shape.empty() && dout.size() == 1) {
        scale *= dout.data[0];
    }

    for (int i = 0; i < total_tokens; ++i) {
        int target_class = cached_targets[i];
        const float* prob_ptr = cached_probs.data.data() + i * V;
        float* dl_ptr = dL.data.data() + i * V;

        for (int v = 0; v < V; ++v) {
            float p = prob_ptr[v];
            if (v == target_class) {
                dl_ptr[v] = (p - 1.0f) * scale;
            } else {
                dl_ptr[v] = p * scale;
            }
        }
    }

    return dL;
}
