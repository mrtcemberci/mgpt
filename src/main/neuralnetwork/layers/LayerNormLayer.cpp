#include "LayerNormLayer.h"

/**
 * Normalising layer, takes in a tensor and produces tensor of same dimension with normalising by both mean and variance.
 */

LayerNormLayer::LayerNormLayer(int channels, float eps)
    : channels(channels),
      eps(eps),
      scale({channels}, 1.0f), // Gamma initialized to 1.0
      shift({channels}, 0.0f) { // Beta initialized to 0.0
}

// FORWARD PASS: Normalize across channel dimension C, apply gamma scale and beta shift
void LayerNormLayer::forward_into(const Tensor& input, Tensor& output) {
    if (input.shape.empty() || input.shape.back() != channels) {
        std::cerr << "LayerNormLayer::forward_into: input channel dimension mismatch!" << std::endl;
        exit(-1);
    }
    cached_input = input;
    input.layer_norm_into(channels, scale, shift, eps, cached_mean, cached_var, cached_x_hat, output);
    cached_output = output;
}

Tensor LayerNormLayer::forward(const Tensor& input) {
    forward_into(input, cached_output);
    return cached_output;
}

// BACKWARD PASS: Compute dGamma, dBeta across all tokens, return dX
void LayerNormLayer::backward_into(const Tensor& dout, Tensor& din) {
    cached_input.layer_norm_backward_into(dout, cached_x_hat, scale, shift, cached_var, eps, din);
    cached_dX = din;
}

Tensor LayerNormLayer::backward(const Tensor& dout) {
    backward_into(dout, cached_dX);
    return cached_dX;
}

std::vector<Tensor*> LayerNormLayer::get_parameters() {
    return { &scale, &shift };
}
