
#include "LinearLayer.h"
#include "Scratchpad.h"
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
    size_t savepoint = 0;
    if (scratchpad && dout.device == Device::CUDA) {
        savepoint = scratchpad->get_savepoint();
    }

    int total_rows = (int)(dout.size() / out_channels);

    Tensor tmp_sum_rows = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({1, out_channels}, scratchpad->get_address((size_t)out_channels), Device::CUDA)
        : cached_sum_rows;
    Tensor tmp_input_T = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({in_channels, total_rows}, scratchpad->get_address((size_t)in_channels * total_rows), Device::CUDA)
        : cached_input_T;
    Tensor tmp_dW = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({in_channels, out_channels}, scratchpad->get_address((size_t)in_channels * out_channels), Device::CUDA)
        : cached_dW;
    Tensor tmp_W_T = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({out_channels, in_channels}, scratchpad->get_address((size_t)out_channels * in_channels), Device::CUDA)
        : cached_W_T;

    dout.sum_rows_into(tmp_sum_rows);
    biases.add_grad(tmp_sum_rows);

    std::vector<int> orig_in_shape = cached_input.shape;
    std::vector<int> orig_dout_shape = dout.shape;
    cached_input.shape = {total_rows, in_channels};
    const_cast<Tensor&>(dout).shape = {total_rows, out_channels};

    cached_input.transpose_into(0, 1, tmp_input_T);
    tmp_input_T.matmul_into(dout, tmp_dW);
    weights.add_grad(tmp_dW);

    weights.transpose_into(0, 1, tmp_W_T);
    dout.matmul_into(tmp_W_T, din);

    cached_input.shape = orig_in_shape;
    const_cast<Tensor&>(dout).shape = orig_dout_shape;
    din.shape = orig_in_shape;
    cached_dX = din;

    if (scratchpad && dout.device == Device::CUDA) {
        scratchpad->restore_savepoint(savepoint);
    }
}

Tensor LinearLayer::backward(const Tensor& dout) {
    backward_into(dout, cached_dX);
    return cached_dX;
}

std::vector<Tensor*> LinearLayer::get_parameters() {
    return { &weights, &biases };
}
