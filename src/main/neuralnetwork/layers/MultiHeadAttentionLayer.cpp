#include "MultiHeadAttentionLayer.h"
#include "cuda_ops.h"
#include "Scratchpad.h"

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

    int max_rope_len = 2048;
    int half_dim = head_dim / 2;
    std::vector<float> h_cos(max_rope_len * half_dim);
    std::vector<float> h_sin(max_rope_len * half_dim);
    for (int t = 0; t < max_rope_len; ++t) {
        for (int j = 0; j < half_dim; ++j) {
            float freq = 1.0f / std::pow(10000.0f, (2.0f * j) / (float)head_dim);
            float angle = t * freq;
            h_cos[t * half_dim + j] = std::cos(angle);
            h_sin[t * half_dim + j] = std::sin(angle);
        }
    }
    rope_cos_table = Tensor({max_rope_len, half_dim}, 0.0f, Device::CPU);
    rope_cos_table.data = h_cos;
    rope_sin_table = Tensor({max_rope_len, half_dim}, 0.0f, Device::CPU);
    rope_sin_table.data = h_sin;
}

// ----------------------------------------------------------------------------
// FORWARD PASS
// ----------------------------------------------------------------------------
void MultiHeadAttentionLayer::forward_into(const Tensor& input, Tensor& output) {
    if (input.device == Device::CUDA && rope_cos_table.device != Device::CUDA) {
        rope_cos_table.to(Device::CUDA);
        rope_sin_table.to(Device::CUDA);
    }

    size_t savepoint = 0;
    if (scratchpad && input.device == Device::CUDA) {
        savepoint = scratchpad->get_savepoint();
    }

    cached_input = input;
    int B = input.shape[0];
    int T = input.shape[1];

    // 1. Project entire Q, K, V at once in ONE violent GEMM (channels -> 3 * channels)
    W_QKV->forward_into(input, cached_QKV_all_fw);

    // 2. Permute Q, K, V directly into batched head format {B * num_heads, T, head_dim}
    Tensor::permute_qkv_to_heads(cached_QKV_all_fw, cached_Q, cached_K, cached_V, B, T, num_heads, head_dim);

    // 2.5 Apply RoPE rotary positional embeddings to Q and K
    Tensor::apply_rope_inplace(cached_Q, rope_cos_table, rope_sin_table, B, num_heads, T, head_dim, true);
    Tensor::apply_rope_inplace(cached_K, rope_cos_table, rope_sin_table, B, num_heads, T, head_dim, true);

    float scale = 1.0f / std::sqrt((float)head_dim);

    Tensor tmp_K_T = (scratchpad && input.device == Device::CUDA)
        ? Tensor::view({B * num_heads, head_dim, T}, scratchpad->get_address((size_t)B * num_heads * head_dim * T), Device::CUDA)
        : cached_K_T;
    Tensor tmp_scores = (scratchpad && input.device == Device::CUDA)
        ? Tensor::view({B * num_heads, T, T}, scratchpad->get_address((size_t)B * num_heads * T * T), Device::CUDA)
        : cached_scores;

    // 3. Batched Multi-Head Attention math across all heads simultaneously
    cached_K.transpose_into(1, 2, tmp_K_T);
    cached_Q.matmul_into(tmp_K_T, tmp_scores);
    tmp_scores.mul_scalar_in_place(scale);
    tmp_scores.causal_mask();
    tmp_scores.softmax_into(-1, cached_probs);
    cached_probs.matmul_into(cached_V, cached_head_contexts);

    // 4. Permute head contexts back to concatenated format {B, T, channels}
    Tensor::permute_heads_to_concat(cached_head_contexts, cached_concat_ctx, B, T, num_heads, head_dim);

    // 5. Output projection W_O
    W_O->forward_into(cached_concat_ctx, output);
    cached_output = output;

    if (scratchpad && input.device == Device::CUDA) {
        scratchpad->restore_savepoint(savepoint);
    }
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

    size_t savepoint = 0;
    if (scratchpad && dout.device == Device::CUDA) {
        savepoint = scratchpad->get_savepoint();
        cached_d_concat_ctx = Tensor::view({B, T, channels}, scratchpad->get_address(B * T * channels), Device::CUDA);
        cached_d_head_contexts = Tensor::view({B * num_heads, T, head_dim}, scratchpad->get_address(B * num_heads * T * head_dim), Device::CUDA);
        cached_probs_T = Tensor::view({B * num_heads, T, T}, scratchpad->get_address(B * num_heads * T * T), Device::CUDA);
        cached_dV = Tensor::view({B * num_heads, T, head_dim}, scratchpad->get_address(B * num_heads * T * head_dim), Device::CUDA);
        cached_V_T = Tensor::view({B * num_heads, head_dim, T}, scratchpad->get_address(B * num_heads * head_dim * T), Device::CUDA);
        cached_dP = Tensor::view({B * num_heads, T, T}, scratchpad->get_address(B * num_heads * T * T), Device::CUDA);
        cached_dS = Tensor::view({B * num_heads, T, T}, scratchpad->get_address(B * num_heads * T * T), Device::CUDA);
        cached_dQ = Tensor::view({B * num_heads, T, head_dim}, scratchpad->get_address(B * num_heads * T * head_dim), Device::CUDA);
        cached_dS_scaled_T = Tensor::view({B * num_heads, T, T}, scratchpad->get_address(B * num_heads * T * T), Device::CUDA);
        cached_dK = Tensor::view({B * num_heads, T, head_dim}, scratchpad->get_address(B * num_heads * T * head_dim), Device::CUDA);
        cached_dQKV_all = Tensor::view({B, T, 3 * channels}, scratchpad->get_address(B * T * 3 * channels), Device::CUDA);
    }

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

    // Apply inverse RoPE rotation to dQ and dK during backprop
    Tensor::apply_rope_inplace(cached_dQ, rope_cos_table, rope_sin_table, B, num_heads, T, head_dim, false);
    Tensor::apply_rope_inplace(cached_dK, rope_cos_table, rope_sin_table, B, num_heads, T, head_dim, false);

    Tensor::permute_heads_grad_to_qkv(cached_dQ, cached_dK, cached_dV, cached_dQKV_all, B, T, num_heads, head_dim);
    W_QKV->backward_into(cached_dQKV_all, din);
    cached_dX = din;

    if (scratchpad && dout.device == Device::CUDA) {
        scratchpad->restore_savepoint(savepoint);
    }
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

void MultiHeadAttentionLayer::set_scratchpad(Scratchpad* pad) {
    this->scratchpad = pad;
    if (W_QKV) W_QKV->set_scratchpad(pad);
    if (W_O) W_O->set_scratchpad(pad);
}