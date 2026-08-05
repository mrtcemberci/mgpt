#include "SingleHeadAttentionLayer.h"
#include "Scratchpad.h"
/**
 * Uses Q, K, and V linear projections to compute Scaled Dot-Product Attention
 *
 * Projects the gathered context back to channel space [C] 
 *    via W_O for addition into the residual stream
 */

SingleHeadAttentionLayer::SingleHeadAttentionLayer(int channels)
    : channels(channels),
      W_Q(channels, channels),
      W_K(channels, channels),
      W_V(channels, channels),
      W_O(channels, channels) {
}

void SingleHeadAttentionLayer::forward_into(const Tensor& input, Tensor& output) {
    size_t savepoint = 0;
    int B = input.shape[0];
    int T = input.shape[1];

    if (scratchpad && input.device == Device::CUDA) {
        savepoint = scratchpad->get_savepoint();
        cached_K_T = Tensor::view({B, channels, T}, scratchpad->get_address(B * channels * T), Device::CUDA);
        cached_scores = Tensor::view({B, T, T}, scratchpad->get_address(B * T * T), Device::CUDA);
        cached_context = Tensor::view({B, T, channels}, scratchpad->get_address(B * T * channels), Device::CUDA);
    }

    cached_input = input;

    W_Q.forward_into(input, cached_Q); // Shape {B, T, C}
    W_K.forward_into(input, cached_K); // Shape {B, T, C}
    W_V.forward_into(input, cached_V); // Shape {B, T, C}

    // Compute Attention Scores: S = (Q * K^T) / sqrt(C)
    cached_K.transpose_into(1, 2, cached_K_T);
    cached_Q.matmul_into(cached_K_T, cached_scores);
    float scale = 1.0f / std::sqrt((float)channels);
    cached_scores = cached_scores * scale;

    // Apply Causal Lower-Triangular Mask (set future positions where col > row to -1e15)
    cached_scores.causal_mask();

    // Stable Softmax along the last dimension (T)
    cached_scores.softmax_into(-1, cached_probs);

    // Weighted combination of Values & Output Projection
    cached_probs.matmul_into(cached_V, cached_context);
    W_O.forward_into(cached_context, output);
    cached_output = output;

    if (scratchpad && input.device == Device::CUDA) {
        scratchpad->restore_savepoint(savepoint);
    }
}

Tensor SingleHeadAttentionLayer::forward(const Tensor& input) {
    forward_into(input, cached_output);
    return cached_output;
}

void SingleHeadAttentionLayer::backward_into(const Tensor& dout, Tensor& din) {
    int B = cached_input.shape[0];
    int T = cached_input.shape[1];
    size_t savepoint = 0;

    if (scratchpad && dout.device == Device::CUDA) {
        savepoint = scratchpad->get_savepoint();
        cached_d_context = Tensor::view({B, T, channels}, scratchpad->get_address(B * T * channels), Device::CUDA);
        cached_dV = Tensor::view({B, T, channels}, scratchpad->get_address(B * T * channels), Device::CUDA);
        cached_dP = Tensor::view({B, T, T}, scratchpad->get_address(B * T * T), Device::CUDA);
        cached_dS = Tensor::view({B, T, T}, scratchpad->get_address(B * T * T), Device::CUDA);
        cached_dS_scaled = Tensor::view({B, T, T}, scratchpad->get_address(B * T * T), Device::CUDA);
        cached_dQ = Tensor::view({B, T, channels}, scratchpad->get_address(B * T * channels), Device::CUDA);
        cached_dK = Tensor::view({B, T, channels}, scratchpad->get_address(B * T * channels), Device::CUDA);
    }

    W_O.backward_into(dout, cached_d_context); // Shape {B, T, C}

    Tensor probs_T = cached_probs.transpose(1, 2);
    probs_T.matmul_into(cached_d_context, cached_dV); // {B, T, T}^T * {B, T, C} -> {B, T, C}

    Tensor V_T = cached_V.transpose(1, 2);
    cached_d_context.matmul_into(V_T, cached_dP);     // {B, T, C} * {B, C, T} -> {B, T, T}

    // Backprop through Softmax along rows of P
    cached_probs.softmax_backward_into(cached_dP, cached_dS);

    // Scale dS by 1 / sqrt(C) and backprop through S = Q * K^T
    float scale = 1.0f / std::sqrt((float)channels);
    cached_dS_scaled = cached_dS * scale; // {B, T, T}

    cached_dS_scaled.matmul_into(cached_K, cached_dQ); // {B, T, T} * {B, T, C} -> {B, T, C}
    Tensor dS_scaled_T = cached_dS_scaled.transpose(1, 2);
    dS_scaled_T.matmul_into(cached_Q, cached_dK);      // {B, T, T}^T * {B, T, C} -> {B, T, C}

    // Backprop through Linear Projections
    W_Q.backward_into(cached_dQ, cached_dX_Q);
    W_K.backward_into(cached_dK, cached_dX_K);
    W_V.backward_into(cached_dV, cached_dX_V);

    // Combine gradients from Query, Key, and Value branches
    if (din.shape != cached_dX_Q.shape || din.device != cached_dX_Q.device || (!din.cuda_data && cached_dX_Q.device == Device::CUDA)) {
        din = Tensor(cached_dX_Q.shape, 0.0f, cached_dX_Q.device);
    }
    din = cached_dX_Q + cached_dX_K + cached_dX_V;
    cached_dX = din;

    if (scratchpad && dout.device == Device::CUDA) {
        scratchpad->restore_savepoint(savepoint);
    }
}

Tensor SingleHeadAttentionLayer::backward(const Tensor& dout) {
    backward_into(dout, cached_dX);
    return cached_dX;
}

std::vector<Tensor*> SingleHeadAttentionLayer::get_parameters() {
    std::vector<Tensor*> params;
    auto q_params = W_Q.get_parameters();
    auto k_params = W_K.get_parameters();
    auto v_params = W_V.get_parameters();
    auto o_params = W_O.get_parameters();

    params.insert(params.end(), q_params.begin(), q_params.end());
    params.insert(params.end(), k_params.begin(), k_params.end());
    params.insert(params.end(), v_params.begin(), v_params.end());
    params.insert(params.end(), o_params.begin(), o_params.end());
    return params;
}

void SingleHeadAttentionLayer::set_scratchpad(Scratchpad* pad) {
    this->scratchpad = pad;
    W_Q.set_scratchpad(pad);
    W_K.set_scratchpad(pad);
    W_V.set_scratchpad(pad);
    W_O.set_scratchpad(pad);
}

ScratchpadFootprint SingleHeadAttentionLayer::get_footprint(int B, int T) { return {(size_t)(4 * B * T * channels + 2 * B * T * T), 0, (size_t)(4 * B * T * channels + 3 * B * T * T)}; }
