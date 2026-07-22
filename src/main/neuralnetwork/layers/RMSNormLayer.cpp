#include "RMSNormLayer.h"

/**
 * RMS Normalising layer, takes in a tensor and produces tensor of same dimension with normalising by RMS.
 */

RMSNormLayer::RMSNormLayer(int channels, float eps)
    : channels(channels),
      eps(eps),
      scale({channels}, 1.0f) { // Gamma initialized to 1.0
}

// FORWARD PASS: Root mean square normalization across channel dimension C, apply gamma scale
void RMSNormLayer::forward_into(const Tensor& input, Tensor& output) {
    if (input.shape.empty() || input.shape.back() != channels) {
        std::cerr << "RMSNormLayer::forward_into: input channel dimension mismatch!" << std::endl;
        exit(-1);
    }
    cached_input = input;
    input.rms_norm_into(channels, scale, eps, cached_rsqrt, cached_x_hat, output);
    cached_output = output;
}

Tensor RMSNormLayer::forward(const Tensor& input) {
    forward_into(input, cached_output);
    return cached_output;
}

// BACKWARD PASS: Compute dGamma across all tokens, return dX
void RMSNormLayer::backward_into(const Tensor& dout, Tensor& din) {
    cached_input.rms_norm_backward_into(dout, cached_x_hat, scale, cached_rsqrt, din);
    cached_dX = din;
}

Tensor RMSNormLayer::backward(const Tensor& dout) {
    backward_into(dout, cached_dX);
    return cached_dX;
}

std::vector<Tensor*> RMSNormLayer::get_parameters() {
    return { &scale };
}
