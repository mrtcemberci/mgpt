#ifndef SINGLEHEADATTENTIONLAYER_H
#define SINGLEHEADATTENTIONLAYER_H

#include "Layer.h"
#include "LinearLayer.h"
#include <cmath>
#include <vector>
#include <iostream>

class SingleHeadAttentionLayer : public Layer {
public: // Public for optimizer updates and inspection
    int channels;
    LinearLayer W_Q; // Query projection: channels -> channels
    LinearLayer W_K; // Key projection:   channels -> channels
    LinearLayer W_V; // Value projection: channels -> channels
    LinearLayer W_O; // Output projection: channels -> channels

private: // Private cached states for backpropagation
    Tensor cached_input;
    Tensor cached_Q;
    Tensor cached_K;
    Tensor cached_V;
    Tensor cached_scores; // Raw attention scores before Softmax
    Tensor cached_probs;  // Attention weights P after Softmax & causal masking

public:
    explicit SingleHeadAttentionLayer(int channels);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& dout) override;
    std::vector<Tensor*> get_parameters() override;
};

#endif //SINGLEHEADATTENTIONLAYER_H
