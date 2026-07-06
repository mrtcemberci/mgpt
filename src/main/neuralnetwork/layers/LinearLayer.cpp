
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
Tensor LinearLayer::forward(const Tensor& input) {

    cached_input = input;

    Tensor output = input.matmul(weights);

    return output + biases;
}

// BACKWARD PASS: Compute dW, db, and return dX
Tensor LinearLayer::backward(const Tensor& dout) {

    int total_rows = (int)(dout.size() / out_channels);

    for (int i = 0; i < total_rows; ++i) {
        for (int j = 0; j < out_channels; ++j) {
            biases.grad[j] += dout.data[i * out_channels + j];
        }
    }

    Tensor X_2d = cached_input.reshape({total_rows, in_channels});
    Tensor dout_2d = dout.reshape({total_rows, out_channels});

    Tensor dW = X_2d.transpose(0, 1).matmul(dout_2d);

    for (size_t i = 0; i < weights.grad.size(); ++i) {
        weights.grad[i] += dW.data[i];
    }

    Tensor W_T = weights.transpose(0, 1);
    Tensor dX = dout.matmul(W_T);

    return dX;
}

std::vector<Tensor*> LinearLayer::get_parameters() {
    return { &weights, &biases };
}
