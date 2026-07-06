#include "LayerNormLayer.h"

LayerNormLayer::LayerNormLayer(int channels, float eps)
    : channels(channels),
      eps(eps),
      scale({channels}, 1.0f), // Gamma initialized to 1.0
      shift({channels}, 0.0f) { // Beta initialized to 0.0
}

// FORWARD PASS: Normalize across channel dimension C, apply gamma scale and beta shift
Tensor LayerNormLayer::forward(const Tensor& input) {
    if (input.shape.empty() || input.shape.back() != channels) {
        std::cerr << "LayerNormLayer::forward: input channel dimension mismatch!" << std::endl;
        exit(-1);
    }
    cached_input = input;
    return input.layer_norm(channels, scale, shift, eps, cached_mean, cached_var, cached_x_hat);
}

// BACKWARD PASS: Compute dGamma, dBeta across all tokens, return dX
Tensor LayerNormLayer::backward(const Tensor& dout) {
    return cached_input.layer_norm_backward(dout, cached_x_hat, scale, shift, cached_mean, cached_var, eps);
}

std::vector<Tensor*> LayerNormLayer::get_parameters() {
    return { &scale, &shift };
}
