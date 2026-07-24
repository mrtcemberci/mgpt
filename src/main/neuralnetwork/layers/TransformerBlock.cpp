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

TransformerBlock::TransformerBlock(int channels, int num_heads)
    : channels(channels),
      ln1(channels),
      attn(channels, num_heads),
      ln2(channels),
      mlp_gate(channels, 4 * channels),
      mlp_up(channels, 4 * channels),
      mlp_down(4 * channels, channels) {
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
        cached_gate_out = Tensor::view({(int)B, (int)T, (int)C*4}, scratchpad->get_address(B*T*C*4), Device::CUDA);
        cached_up_out = Tensor::view({(int)B, (int)T, (int)C*4}, scratchpad->get_address(B*T*C*4), Device::CUDA);
        cached_gate_swished = Tensor::view({(int)B, (int)T, (int)C*4}, scratchpad->get_address(B*T*C*4), Device::CUDA);
    }

    ln1.forward_into(input, cached_ln1_out);
    attn.forward_into(cached_ln1_out, cached_attn_out);
    Tensor::add_into(input, cached_attn_out, cached_x1);

    ln2.forward_into(cached_x1, cached_ln2_out);

    // SwiGLU: swish(gate(x)) ⊙ up(x)
    mlp_gate.forward_into(cached_ln2_out, cached_gate_out);   // [B, T, 4C] raw gate (persists for swish backward)
    mlp_up.forward_into(cached_ln2_out, cached_up_out);       // [B, T, 4C] up projection (persists for backward)
    cached_gate_out.swish_into(cached_gate_swished);           // [B, T, 4C] swish(gate) (persists for d_up in backward)

    // Temporary buffer for swish(gate) ⊙ up — mlp_down caches its own input, so this doesn't need to persist
    int B = cached_gate_out.shape[0], T = cached_gate_out.shape[1], C4 = cached_gate_out.shape[2];
    size_t fwd_savepoint = 0;
    if (scratchpad && cached_gate_out.device == Device::CUDA) {
        fwd_savepoint = scratchpad->get_savepoint();
    }
    Tensor swiglu_tmp = (scratchpad && cached_gate_out.device == Device::CUDA)
        ? Tensor::view({B, T, C4}, scratchpad->get_address((size_t)B * T * C4), Device::CUDA)
        : cached_swiglu_tmp;
    cached_gate_swished.pairwise_mult_into(cached_up_out, swiglu_tmp);
    mlp_down.forward_into(swiglu_tmp, cached_out);
    if (scratchpad && cached_gate_out.device == Device::CUDA) {
        scratchpad->restore_savepoint(fwd_savepoint);
    }

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
    Tensor tmp_d_down = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({B, T, 4 * C}, scratchpad->get_address((size_t)B * T * 4 * C), Device::CUDA)
        : cached_d_down;
    Tensor tmp_d_gate = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({B, T, 4 * C}, scratchpad->get_address((size_t)B * T * 4 * C), Device::CUDA)
        : cached_d_gate;
    Tensor tmp_d_up = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({B, T, 4 * C}, scratchpad->get_address((size_t)B * T * 4 * C), Device::CUDA)
        : cached_d_up;
    Tensor tmp_d_swish = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({B, T, 4 * C}, scratchpad->get_address((size_t)B * T * 4 * C), Device::CUDA)
        : cached_d_swish;
    Tensor tmp_d_gate_proj = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({B, T, C}, scratchpad->get_address((size_t)B * T * C), Device::CUDA)
        : cached_d_gate_proj;
    Tensor tmp_d_up_proj = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({B, T, C}, scratchpad->get_address((size_t)B * T * C), Device::CUDA)
        : cached_d_up_proj;
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

    // Backprop through SwiGLU MLP branch: down -> (swish_backward ⊙ multiply) -> gate/up -> ln2
    mlp_down.backward_into(dout, tmp_d_down);                                      // d_down = [B,T,4C]
    tmp_d_down.pairwise_mult_into(cached_gate_swished, tmp_d_up);                  // d_up = d_down ⊙ swish(gate)
    tmp_d_down.pairwise_mult_into(cached_up_out, tmp_d_swish);                     // d_swish_in = d_down ⊙ up
    Tensor::swish_backward_into(cached_gate_out, tmp_d_swish, tmp_d_gate);         // d_gate = swish'(gate_pre) * d_swish_in
    mlp_gate.backward_into(tmp_d_gate, tmp_d_gate_proj);                           // d_gate_proj = [B,T,C]
    mlp_up.backward_into(tmp_d_up, tmp_d_up_proj);                                 // d_up_proj = [B,T,C]
    Tensor::add_into(tmp_d_gate_proj, tmp_d_up_proj, tmp_d_ln2);                   // merge gate + up gradients
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
    auto p_gate = mlp_gate.get_parameters(); params.insert(params.end(), p_gate.begin(), p_gate.end());
    auto p_up = mlp_up.get_parameters(); params.insert(params.end(), p_up.begin(), p_up.end());
    auto p_down = mlp_down.get_parameters(); params.insert(params.end(), p_down.begin(), p_down.end());
    return params;
}
