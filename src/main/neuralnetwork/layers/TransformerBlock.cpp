#include "TransformerBlock.h"

TransformerBlock::TransformerBlock(int channels)
    : channels(channels),
      ln1(channels),
      attn(channels),
      ln2(channels),
      mlp_fc1(channels, 4 * channels),
      act(),
      mlp_fc2(4 * channels, channels) {
}

// FORWARD PASS: Pre-Norm Attention + Skip -> Pre-Norm MLP + Skip
Tensor TransformerBlock::forward(const Tensor& input) {
    cached_input = input;

    cached_ln1_out = ln1.forward(input);
    cached_attn_out = attn.forward(cached_ln1_out);
    cached_x1 = input + cached_attn_out;

    cached_ln2_out = ln2.forward(cached_x1);
    cached_fc1_out = mlp_fc1.forward(cached_ln2_out);
    cached_act_out = act.forward(cached_fc1_out);
    Tensor fc2_out = mlp_fc2.forward(cached_act_out);

    return cached_x1 + fc2_out;
}

// BACKWARD PASS: Chain rule backwards through both skip connection branches
Tensor TransformerBlock::backward(const Tensor& dout) {
    // Backprop through MLP branch: fc2 -> act -> fc1 -> ln2
    Tensor d_fc2 = mlp_fc2.backward(dout);
    Tensor d_act = act.backward(d_fc2);
    Tensor d_fc1 = mlp_fc1.backward(d_act);
    Tensor d_ln2 = ln2.backward(d_fc1);

    // Gradient entering X1 is the sum of direct skip connection (dout) and MLP branch (d_ln2)
    Tensor d_x1 = dout + d_ln2;

    // Backprop through Self-Attention branch: attn -> ln1
    Tensor d_attn = attn.backward(d_x1);
    Tensor d_ln1 = ln1.backward(d_attn);

    // Gradient entering initial input X is sum of direct skip connection (d_x1) and Attn branch (d_ln1)
    return d_x1 + d_ln1;
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
