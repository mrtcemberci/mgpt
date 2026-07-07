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

private: // Private cached states and pre-allocated workspaces for forward/backward execution
    Tensor cached_input;
    std::vector<Tensor> cached_qkv_slices;
    std::vector<Tensor> cached_Q_heads;
    std::vector<Tensor> cached_K_heads;
    std::vector<Tensor> cached_V_heads;
    std::vector<Tensor> cached_K_T_heads;
    std::vector<Tensor> cached_scores_heads;
    std::vector<Tensor> cached_probs_heads; // Attention weights P after Softmax & causal masking
    std::vector<Tensor> cached_head_contexts;
    Tensor cached_concat_ctx;
    Tensor cached_QKV_all_fw;
    std::vector<Tensor> cached_d_head_contexts;
    std::vector<Tensor> cached_probs_heads_T;
    std::vector<Tensor> cached_V_heads_T;
    std::vector<Tensor> cached_dV_heads;
    std::vector<Tensor> cached_dP_heads;
    std::vector<Tensor> cached_dS_heads;
    std::vector<Tensor> cached_dS_scaled_heads;
    std::vector<Tensor> cached_dS_scaled_T;
    std::vector<Tensor> cached_dQ_heads;
    std::vector<Tensor> cached_dK_heads;
    Tensor cached_dQ_all;
    Tensor cached_dK_all;
    Tensor cached_dV_all;
    Tensor cached_dQKV_all;
    Tensor cached_d_concat_ctx;
    Tensor cached_output;
    Tensor cached_dX;

public:
    MultiHeadAttentionLayer(int channels, int num_heads);

    Tensor forward(const Tensor& input) override;
    void forward_into(const Tensor& input, Tensor& output) override;
    Tensor backward(const Tensor& dout) override;
    void backward_into(const Tensor& dout, Tensor& din) override;
    std::vector<Tensor*> get_parameters() override;
};

#endif // MULTIHEADATTENTIONLAYER_H