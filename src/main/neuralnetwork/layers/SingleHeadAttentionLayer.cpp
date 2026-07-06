#include "SingleHeadAttentionLayer.h"

SingleHeadAttentionLayer::SingleHeadAttentionLayer(int channels)
    : channels(channels),
      W_Q(channels, channels),
      W_K(channels, channels),
      W_V(channels, channels),
      W_O(channels, channels) {
}

// ----------------------------------------------------------------------------
// FORWARD PASS: Projections -> Scaled Dot-Product -> Causal Mask -> Softmax -> Output
// ----------------------------------------------------------------------------
Tensor SingleHeadAttentionLayer::forward(const Tensor& input) {
    cached_input = input;

    // 1. Linear Projections for Queries, Keys, and Values
    cached_Q = W_Q.forward(input); // Shape {B, T, C}
    cached_K = W_K.forward(input); // Shape {B, T, C}
    cached_V = W_V.forward(input); // Shape {B, T, C}

    // 2. Compute Attention Scores: S = (Q * K^T) / sqrt(C)
    float scale = 1.0f / std::sqrt((float)channels);
    cached_scores = cached_Q.matmul(cached_K.transpose(1, 2)) * scale; // Shape {B, T, T}

    // 3. Apply Causal Lower-Triangular Mask (set future positions where col > row to -1e15)
    cached_scores.causal_mask();

    // 4. Stable Softmax along the last dimension (T)
    cached_probs = cached_scores.softmax(-1);

    // 5. Weighted combination of Values & Output Projection
    Tensor context = cached_probs.matmul(cached_V); // {B, T, T} * {B, T, C} -> {B, T, C}
    return W_O.forward(context);
}

// ----------------------------------------------------------------------------
// BACKWARD PASS: Chain rule backwards through W_O, Softmax, Matmuls, and W_Q/W_K/W_V
// ----------------------------------------------------------------------------
Tensor SingleHeadAttentionLayer::backward(const Tensor& dout) {
    // 1. Backprop through Output Projection W_O
    Tensor d_context = W_O.backward(dout); // Shape {B, T, C}

    // 2. Backprop through context = P * V
    Tensor dV = cached_probs.transpose(1, 2).matmul(d_context); // {B, T, T}^T * {B, T, C} -> {B, T, C}
    Tensor dP = d_context.matmul(cached_V.transpose(1, 2));     // {B, T, C} * {B, C, T} -> {B, T, T}

    // 3. Backprop through Softmax along rows of P
    Tensor dS = cached_probs.softmax_backward(dP);

    // 4. Scale dS by 1 / sqrt(C) and backprop through S = Q * K^T
    float scale = 1.0f / std::sqrt((float)channels);
    Tensor dS_scaled = dS * scale; // {B, T, T}

    Tensor dQ = dS_scaled.matmul(cached_K);                 // {B, T, T} * {B, T, C} -> {B, T, C}
    Tensor dK = dS_scaled.transpose(1, 2).matmul(cached_Q); // {B, T, T}^T * {B, T, C} -> {B, T, C}

    // 5. Backprop through Linear Projections
    Tensor dX_Q = W_Q.backward(dQ);
    Tensor dX_K = W_K.backward(dK);
    Tensor dX_V = W_V.backward(dV);

    // 6. Combine gradients from Query, Key, and Value branches
    return dX_Q + dX_K + dX_V;
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
