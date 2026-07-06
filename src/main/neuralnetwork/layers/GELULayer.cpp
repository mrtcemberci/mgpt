#include "GELULayer.h"

// GELU FORWARD PASS: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
Tensor GELULayer::forward(const Tensor& input) {
    cached_input = input;
    return input.gelu();
}

// GELU BACKWARD PASS: dX = dout * dy/dx
Tensor GELULayer::backward(const Tensor& dout) {
    return cached_input.gelu_backward(dout);
}
