#include "TransformerBlock.h"
#include "Scratchpad.h"

/**
 * The transformer block is the building block that is repeated in the transformer
 * 
 * Each transformer consists of
 * 
 * Layer norm -> Attention -> Layer norm -> projection to 4*channels, projection down to channels
 * 
 * Uses residual connections
 * 
 * Uses the swiglu activation.
 * 
 * X_next = X + Attention(RMSNorm1(X)) + MLP(RMSNorm2(X + Attention(RMSNorm1(X))))
 */

TransformerBlock::TransformerBlock(int channels, int num_heads, int num_experts, int top_k, bool use_flash_attention) 
    : channels(channels),
      ln1(channels), 
      attn(channels, num_heads, use_flash_attention), 
      ln2(channels), 
      mlp(channels, 4 * channels, num_experts, top_k) {
}

// FORWARD PASS: Pre-Norm Attention + Skip -> Pre-Norm SwiGLU MLP + Skip
void TransformerBlock::forward_into(const Tensor& input, Tensor& output) {
    cached_input = input;

    if (scratchpad && input.device == Device::CUDA) {
        size_t B = input.shape[0], T = input.shape[1], C = input.shape[2];
        cached_ln1_out = Tensor::view(input.shape, scratchpad->get_address(B*T*C), Device::CUDA);
        cached_attn_out = Tensor::view(input.shape, scratchpad->get_address(B*T*C), Device::CUDA);
        cached_x1 = Tensor::view(input.shape, scratchpad->get_address(B*T*C), Device::CUDA);
        cached_ln2_out = Tensor::view(input.shape, scratchpad->get_address(B*T*C), Device::CUDA);
    }

    ln1.forward_into(input, cached_ln1_out);
    attn.forward_into(cached_ln1_out, cached_attn_out);
    Tensor::add_into(input, cached_attn_out, cached_x1);

    ln2.forward_into(cached_x1, cached_ln2_out);

    mlp.forward_into(cached_ln2_out, cached_out);

    Tensor::add_into(cached_x1, cached_out, output);
    cached_out = output;
}

Tensor TransformerBlock::forward(const Tensor& input) {
    forward_into(input, cached_out);
    return cached_out;
}

// BACKWARD PASS: Chain rule backwards through both skip connection branches
void TransformerBlock::backward_into(const Tensor& dout, Tensor& din) {
    size_t savepoint = 0;
    if (scratchpad && dout.device == Device::CUDA) {
        savepoint = scratchpad->get_savepoint();
    }

    int B = dout.shape[0];
    int T = dout.shape[1];
    int C = dout.shape[2];

    // Allocate scratch views for backward intermediates
    Tensor tmp_d_ln2 = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({B, T, C}, scratchpad->get_address((size_t)B * T * C), Device::CUDA)
        : cached_d_ln2;
    Tensor tmp_d_x1 = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({B, T, C}, scratchpad->get_address((size_t)B * T * C), Device::CUDA)
        : cached_d_x1;
    Tensor tmp_d_attn = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({B, T, C}, scratchpad->get_address((size_t)B * T * C), Device::CUDA)
        : cached_d_attn;
    Tensor tmp_d_ln1 = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({B, T, C}, scratchpad->get_address((size_t)B * T * C), Device::CUDA)
        : cached_d_ln1;

        
    mlp.backward_into(dout, tmp_d_ln2);
    ln2.backward_into(tmp_d_ln2, tmp_d_ln2);                                       // reuse buffer for ln2 backward

    // Gradient entering X1 is the sum of direct skip connection (dout) and MLP branch (d_ln2)
    Tensor::add_into(dout, tmp_d_ln2, tmp_d_x1);

    // Backprop through Self-Attention branch: attn -> ln1
    attn.backward_into(tmp_d_x1, tmp_d_attn);
    ln1.backward_into(tmp_d_attn, tmp_d_ln1);

    // Gradient entering initial input X is sum of direct skip connection (d_x1) and Attn branch (d_ln1)
    Tensor::add_into(tmp_d_x1, tmp_d_ln1, din);
    cached_dX = din;

    if (scratchpad && dout.device == Device::CUDA) {
        scratchpad->restore_savepoint(savepoint);
    }
}

Tensor TransformerBlock::backward(const Tensor& dout) {
    backward_into(dout, cached_dX);
    return cached_dX;
}

std::vector<Tensor*> TransformerBlock::get_parameters() {
    std::vector<Tensor*> params;
    auto p_ln1 = ln1.get_parameters(); params.insert(params.end(), p_ln1.begin(), p_ln1.end());
    auto p_attn = attn.get_parameters(); params.insert(params.end(), p_attn.begin(), p_attn.end());
    auto p_ln2 = ln2.get_parameters(); params.insert(params.end(), p_ln2.begin(), p_ln2.end());
    auto p_mlp = mlp.get_parameters(); params.insert(params.end(), p_mlp.begin(), p_mlp.end());
    return params;
}
