
#include "LinearLayer.h"
#include <iostream>
#include <cmath>

LinearLayer::LinearLayer(int in_channels, int out_channels)
    : in_channels(in_channels),
      out_channels(out_channels),
      weights(Tensor::randn({in_channels, out_channels}, 0.0f, std::sqrt(2.0f / (float)in_channels))), // Kaiming normal random init
      biases({1, out_channels}, 0.0f) {            // Initialized with 0 bias
}

// FORWARD PASS: Y = X * W + b
void LinearLayer::forward_into(const Tensor& input, Tensor& output) {
    cached_input = input;
    input.matmul_into(weights, output);
    output.add_broadcast_in_place(biases);
    cached_output = output;
}

Tensor LinearLayer::forward(const Tensor& input) {
    forward_into(input, cached_output);
    return cached_output;
}

// BACKWARD PASS: Compute dW, db, and return dX
void LinearLayer::backward_into(const Tensor& dout, Tensor& din) {
    int total_rows = (int)(dout.size() / out_channels);
    dout.sum_rows_into(cached_sum_rows);
    biases.add_grad(cached_sum_rows);

    std::vector<int> orig_in_shape = cached_input.shape;
    std::vector<int> orig_dout_shape = dout.shape;
    cached_input.shape = {total_rows, in_channels};
    const_cast<Tensor&>(dout).shape = {total_rows, out_channels};

    cached_input.transpose_into(0, 1, cached_input_T);
    cached_input_T.matmul_into(dout, cached_dW);
    weights.add_grad(cached_dW);

    weights.transpose_into(0, 1, cached_W_T);
    dout.matmul_into(cached_W_T, din);

    cached_input.shape = orig_in_shape;
    const_cast<Tensor&>(dout).shape = orig_dout_shape;
    din.shape = orig_in_shape;
    cached_dX = din;
}

Tensor LinearLayer::backward(const Tensor& dout) {
    backward_into(dout, cached_dX);
    return cached_dX;
}

std::vector<Tensor*> LinearLayer::get_parameters() {
    return { &weights, &biases };
}
