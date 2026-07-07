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

    // 1. Project entire Q, K, V at once in ONE violent GEMM (channels -> 3 * channels)
    W_QKV->forward_into(input, cached_QKV_all_fw);

    // 2. Slice the output tensor into three distinct Q, K, and V chunks
    Tensor::slice_qkv_into(cached_QKV_all_fw, channels, cached_qkv_slices);
    Tensor& Q_all = cached_qkv_slices[0];
    Tensor& K_all = cached_qkv_slices[1];
    Tensor& V_all = cached_qkv_slices[2];

    // 3. Split the unified projections into individual heads for attention math
    Tensor::split_channels_into(Q_all, num_heads, cached_Q_heads);
    Tensor::split_channels_into(K_all, num_heads, cached_K_heads);
    Tensor::split_channels_into(V_all, num_heads, cached_V_heads);

    if ((int)cached_K_T_heads.size() != num_heads) cached_K_T_heads.resize(num_heads);
    if ((int)cached_scores_heads.size() != num_heads) cached_scores_heads.resize(num_heads);
    if ((int)cached_probs_heads.size() != num_heads) cached_probs_heads.resize(num_heads);
    if ((int)cached_head_contexts.size() != num_heads) cached_head_contexts.resize(num_heads);

    float scale = 1.0f / std::sqrt((float)head_dim);

    // 4. Compute attention per head
    for (int h = 0; h < num_heads; ++h) {
        Tensor& Q = cached_Q_heads[h];
        Tensor& K = cached_K_heads[h];
        Tensor& V = cached_V_heads[h];

        K.transpose_into(1, 2, cached_K_T_heads[h]);
        Q.matmul_into(cached_K_T_heads[h], cached_scores_heads[h]);
        cached_scores_heads[h].mul_scalar_in_place(scale);
        cached_scores_heads[h].causal_mask();
        cached_scores_heads[h].softmax_into(-1, cached_probs_heads[h]);
        cached_probs_heads[h].matmul_into(V, cached_head_contexts[h]);
    }

    // 5. Re-concatenate the heads back into a single tensor and project
    Tensor::concat_channels_into(cached_head_contexts, cached_concat_ctx);
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
    W_O->backward_into(dout, cached_d_concat_ctx);
    Tensor::split_channels_into(cached_d_concat_ctx, num_heads, cached_d_head_contexts);

    if ((int)cached_dQ_heads.size() != num_heads) {
        cached_dQ_heads.resize(num_heads);
        cached_dK_heads.resize(num_heads);
        cached_dV_heads.resize(num_heads);
        cached_probs_heads_T.resize(num_heads);
        cached_V_heads_T.resize(num_heads);
        cached_dP_heads.resize(num_heads);
        cached_dS_heads.resize(num_heads);
        cached_dS_scaled_heads.resize(num_heads);
        cached_dS_scaled_T.resize(num_heads);
    }

    float scale = 1.0f / std::sqrt((float)head_dim);

    // 1. Calculate gradients per head
    for (int h = 0; h < num_heads; ++h) {
        Tensor& d_head_ctx = cached_d_head_contexts[h];

        cached_probs_heads[h].transpose_into(1, 2, cached_probs_heads_T[h]);
        cached_probs_heads_T[h].matmul_into(d_head_ctx, cached_dV_heads[h]);

        cached_V_heads[h].transpose_into(1, 2, cached_V_heads_T[h]);
        d_head_ctx.matmul_into(cached_V_heads_T[h], cached_dP_heads[h]);

        cached_probs_heads[h].softmax_backward_into(cached_dP_heads[h], cached_dS_heads[h]);

        if (cached_dS_scaled_heads[h].shape != cached_dS_heads[h].shape || cached_dS_scaled_heads[h].device != cached_dS_heads[h].device || !cached_dS_scaled_heads[h].cuda_data) {
            cached_dS_scaled_heads[h] = Tensor(cached_dS_heads[h].shape, 0.0f, cached_dS_heads[h].device);
        }
        if (cached_dS_heads[h].device == Device::CUDA) {
            if (cached_dS_scaled_heads[h].get_data_ptr() != cached_dS_heads[h].get_data_ptr()) {
                cuda_ops::copy_device_to_device(cached_dS_scaled_heads[h].get_data_ptr(), cached_dS_heads[h].get_data_ptr(), cached_dS_heads[h].size());
            }
            cached_dS_scaled_heads[h].mul_scalar_in_place(scale);
        } else {
            cached_dS_scaled_heads[h] = cached_dS_heads[h];
            cached_dS_scaled_heads[h].mul_scalar_in_place(scale);
        }

        cached_dS_scaled_heads[h].matmul_into(cached_K_heads[h], cached_dQ_heads[h]);

        cached_dS_scaled_heads[h].transpose_into(1, 2, cached_dS_scaled_T[h]);
        cached_dS_scaled_T[h].matmul_into(cached_Q_heads[h], cached_dK_heads[h]);
    }

    // 2. Re-concatenate head gradients into full projections
    Tensor::concat_channels_into(cached_dQ_heads, cached_dQ_all);
    Tensor::concat_channels_into(cached_dK_heads, cached_dK_all);
    Tensor::concat_channels_into(cached_dV_heads, cached_dV_all);

    // 3. Concat gradients into dQKV and backward through single massive W_QKV
    Tensor::concat_qkv_grad_into(cached_dQ_all, cached_dK_all, cached_dV_all, cached_dQKV_all);
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