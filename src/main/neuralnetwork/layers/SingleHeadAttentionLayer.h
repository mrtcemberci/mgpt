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
    Tensor cached_K_T;
    Tensor cached_scores; // Raw attention scores before Softmax
    Tensor cached_probs;  // Attention weights P after Softmax & causal masking
    Tensor cached_context;
    Tensor cached_output;
    Tensor cached_d_context;
    Tensor cached_dV;
    Tensor cached_dP;
    Tensor cached_dS;
    Tensor cached_dS_scaled;
    Tensor cached_dQ;
    Tensor cached_dK;
    Tensor cached_dX_Q;
    Tensor cached_dX_K;
    Tensor cached_dX_V;
    Tensor cached_dX;

public:
    explicit SingleHeadAttentionLayer(int channels);

    Tensor forward(const Tensor& input) override;
    void forward_into(const Tensor& input, Tensor& output) override;
    Tensor backward(const Tensor& dout) override;
    void backward_into(const Tensor& dout, Tensor& din) override;
    std::vector<Tensor*> get_parameters() override;
    ScratchpadFootprint get_footprint(int B, int T) override;
    void set_scratchpad(Scratchpad* pad) override;
};

#endif //SINGLEHEADATTENTIONLAYER_H
