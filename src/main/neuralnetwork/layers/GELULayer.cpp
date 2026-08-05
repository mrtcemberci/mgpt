#include "GELULayer.h"

/**
 * Activation layer, takes in BxTxC and applies GELU activation element wise, produces a tensor of same dimension
 */

// GELU FORWARD PASS: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void GELULayer::forward_into(const Tensor& input, Tensor& output) {
    cached_input = input;
    input.gelu_into(output);
    cached_output = output;
}

Tensor GELULayer::forward(const Tensor& input) {
    forward_into(input, cached_output);
    return cached_output;
}

// GELU BACKWARD PASS: dX = dout * dy/dx
void GELULayer::backward_into(const Tensor& dout, Tensor& din) {
    cached_input.gelu_backward_into(dout, cached_d_gelu_workspace, din);
    cached_dX = din;
}

Tensor GELULayer::backward(const Tensor& dout) {
    backward_into(dout, cached_dX);
    return cached_dX;
}

// does not use the scratch pad
ScratchpadFootprint GELULayer::get_footprint(int B, int T) { return {0, 0, 0}; }
