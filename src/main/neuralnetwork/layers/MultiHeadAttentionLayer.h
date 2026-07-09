#ifndef MULTIHEADATTENTIONLAYER_H
#define MULTIHEADATTENTIONLAYER_H

#include "Layer.h"
#include "LinearLayer.h"
#include <vector>
#include <memory>
#include <cmath>
#include <iostream>

class MultiHeadAttentionLayer : public Layer {
public: // Public for optimizer updates and inspection
    int channels;
    int num_heads;
    int head_dim;

    // CONSOLIDATED Linear projection: channels -> channels * 3 (W_QKV smashed together)
    std::unique_ptr<LinearLayer> W_QKV;

    // Final output projection: channels -> channels
    std::unique_ptr<LinearLayer> W_O;

    // Precomputed RoPE trig tables
    Tensor rope_cos_table;
    Tensor rope_sin_table;

private: // Private cached states and pre-allocated workspaces for forward/backward execution
    Tensor cached_input;
    Tensor cached_QKV_all_fw;
    Tensor cached_Q;
    Tensor cached_K;
    Tensor cached_V;
    Tensor cached_K_T;
    Tensor cached_scores;
    Tensor cached_probs;
    Tensor cached_head_contexts;
    Tensor cached_concat_ctx;
    Tensor cached_d_concat_ctx;
    Tensor cached_d_head_contexts;
    Tensor cached_probs_T;
    Tensor cached_V_T;
    Tensor cached_dV;
    Tensor cached_dP;
    Tensor cached_dS;
    Tensor cached_dS_scaled_T;
    Tensor cached_dQ;
    Tensor cached_dK;
    Tensor cached_dQKV_all;
    Tensor cached_output;
    Tensor cached_dX;

public:
    MultiHeadAttentionLayer(int channels, int num_heads);

    Tensor forward(const Tensor& input) override;
    void forward_into(const Tensor& input, Tensor& output) override;
    Tensor backward(const Tensor& dout) override;
    void backward_into(const Tensor& dout, Tensor& din) override;
    std::vector<Tensor*> get_parameters() override;
    void set_scratchpad(Scratchpad* pad) override;
};

#endif // MULTIHEADATTENTIONLAYER_H