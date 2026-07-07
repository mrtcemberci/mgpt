#include "MultiHeadAttentionLayer.h"
#include "cuda_ops.h"

MultiHeadAttentionLayer::MultiHeadAttentionLayer(int channels, int num_heads_requested)
    : channels(channels) {
    num_heads = num_heads_requested;
    // Safety check: gracefully fallback if channels don't divide evenly
    if (channels % num_heads != 0) {
        while (num_heads > 1 && channels % num_heads != 0) {
            num_heads--;
        }
    }
    head_dim = channels / num_heads;

    // CONSOLIDATED PROJECTIONS: 1 massive kernel projecting channels -> 3 * channels
    W_QKV = std::make_unique<LinearLayer>(channels, 3 * channels);
    W_O = std::make_unique<LinearLayer>(channels, channels);
}

// ----------------------------------------------------------------------------
// FORWARD PASS
// ----------------------------------------------------------------------------
void MultiHeadAttentionLayer::forward_into(const Tensor& input, Tensor& output) {
    cached_input = input;
    int B = input.shape[0];
    int T = input.shape[1];

    // 1. Project entire Q, K, V at once in ONE violent GEMM (channels -> 3 * channels)
    W_QKV->forward_into(input, cached_QKV_all_fw);

    // 2. Permute Q, K, V directly into batched head format {B * num_heads, T, head_dim}
    Tensor::permute_qkv_to_heads(cached_QKV_all_fw, cached_Q, cached_K, cached_V, B, T, num_heads, head_dim);

    float scale = 1.0f / std::sqrt((float)head_dim);

    // 3. Batched Multi-Head Attention math across all heads simultaneously
    cached_K.transpose_into(1, 2, cached_K_T);
    cached_Q.matmul_into(cached_K_T, cached_scores);
    cached_scores.mul_scalar_in_place(scale);
    cached_scores.causal_mask();
    cached_scores.softmax_into(-1, cached_probs);
    cached_probs.matmul_into(cached_V, cached_head_contexts);

    // 4. Permute head contexts back to concatenated format {B, T, channels}
    Tensor::permute_heads_to_concat(cached_head_contexts, cached_concat_ctx, B, T, num_heads, head_dim);

    // 5. Output projection W_O
    W_O->forward_into(cached_concat_ctx, output);
    cached_output = output;
}

Tensor MultiHeadAttentionLayer::forward(const Tensor& input) {
    forward_into(input, cached_output);
    return cached_output;
}

// ----------------------------------------------------------------------------
// BACKWARD PASS
// ----------------------------------------------------------------------------
void MultiHeadAttentionLayer::backward_into(const Tensor& dout, Tensor& din) {
    int B = cached_input.shape[0];
    int T = cached_input.shape[1];

    W_O->backward_into(dout, cached_d_concat_ctx);
    Tensor::permute_concat_to_heads(cached_d_concat_ctx, cached_d_head_contexts, B, T, num_heads, head_dim);

    float scale = 1.0f / std::sqrt((float)head_dim);

    cached_probs.transpose_into(1, 2, cached_probs_T);
    cached_probs_T.matmul_into(cached_d_head_contexts, cached_dV);

    cached_V.transpose_into(1, 2, cached_V_T);
    cached_d_head_contexts.matmul_into(cached_V_T, cached_dP);

    cached_probs.softmax_backward_into(cached_dP, cached_dS);

    cached_dS.mul_scalar_in_place(scale);

    cached_dS.matmul_into(cached_K, cached_dQ);

    cached_dS.transpose_into(1, 2, cached_dS_scaled_T);
    cached_dS_scaled_T.matmul_into(cached_Q, cached_dK);

    Tensor::permute_heads_grad_to_qkv(cached_dQ, cached_dK, cached_dV, cached_dQKV_all, B, T, num_heads, head_dim);
    W_QKV->backward_into(cached_dQKV_all, din);
    cached_dX = din;
}

Tensor MultiHeadAttentionLayer::backward(const Tensor& dout) {
    backward_into(dout, cached_dX);
    return cached_dX;
}

std::vector<Tensor*> MultiHeadAttentionLayer::get_parameters() {
    std::vector<Tensor*> params;
    if (W_QKV) {
        auto qkv_params = W_QKV->get_parameters();
        params.insert(params.end(), qkv_params.begin(), qkv_params.end());
    }
    if (W_O) {
        auto o_params = W_O->get_parameters();
        params.insert(params.end(), o_params.begin(), o_params.end());
    }
    return params;
}