#include "TransformerBlock.h"

TransformerBlock::TransformerBlock(int channels, int num_heads)
    : channels(channels),
      ln1(channels),
      attn(channels, num_heads),
      ln2(channels),
      mlp_fc1(channels, 4 * channels),
      act(),
      mlp_fc2(4 * channels, channels) {
}

// FORWARD PASS: Pre-Norm Attention + Skip -> Pre-Norm MLP + Skip
void TransformerBlock::forward_into(const Tensor& input, Tensor& output) {
    cached_input = input;

    ln1.forward_into(input, cached_ln1_out);
    attn.forward_into(cached_ln1_out, cached_attn_out);
    Tensor::add_into(input, cached_attn_out, cached_x1);

    ln2.forward_into(cached_x1, cached_ln2_out);
    mlp_fc1.forward_into(cached_ln2_out, cached_fc1_out);
    act.forward_into(cached_fc1_out, cached_act_out);
    mlp_fc2.forward_into(cached_act_out, cached_fc2_out);

    Tensor::add_into(cached_x1, cached_fc2_out, output);
    cached_out = output;
}

Tensor TransformerBlock::forward(const Tensor& input) {
    forward_into(input, cached_out);
    return cached_out;
}

// BACKWARD PASS: Chain rule backwards through both skip connection branches
void TransformerBlock::backward_into(const Tensor& dout, Tensor& din) {
    // Backprop through MLP branch: fc2 -> act -> fc1 -> ln2
    mlp_fc2.backward_into(dout, cached_d_fc2);
    act.backward_into(cached_d_fc2, cached_d_act);
    mlp_fc1.backward_into(cached_d_act, cached_d_fc1);
    ln2.backward_into(cached_d_fc1, cached_d_ln2);

    // Gradient entering X1 is the sum of direct skip connection (dout) and MLP branch (d_ln2)
    Tensor::add_into(dout, cached_d_ln2, cached_d_x1);

    // Backprop through Self-Attention branch: attn -> ln1
    attn.backward_into(cached_d_x1, cached_d_attn);
    ln1.backward_into(cached_d_attn, cached_d_ln1);

    // Gradient entering initial input X is sum of direct skip connection (d_x1) and Attn branch (d_ln1)
    Tensor::add_into(cached_d_x1, cached_d_ln1, din);
    cached_dX = din;
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
    auto p_fc1 = mlp_fc1.get_parameters(); params.insert(params.end(), p_fc1.begin(), p_fc1.end());
    auto p_fc2 = mlp_fc2.get_parameters(); params.insert(params.end(), p_fc2.begin(), p_fc2.end());
    return params;
}
