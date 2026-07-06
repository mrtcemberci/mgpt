#include "GELULayer.h"

// GELU FORWARD PASS: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
Tensor GELULayer::forward(const Tensor& input) {
    cached_input = input;

    return input.map([](float x) {
        float x3 = x * x * x;
        float s = 0.7978845608f * (x + 0.044715f * x3); // sqrt(2/pi) ≈ 0.7978845608f
        return 0.5f * x * (1.0f + std::tanh(s));
    });
}

// GELU BACKWARD PASS: dX = dout * dy/dx
Tensor GELULayer::backward(const Tensor& dout) {
    Tensor d_gelu = cached_input.map([](float x) {
        float x2 = x * x;
        float x3 = x2 * x;
        float s = 0.7978845608f * (x + 0.044715f * x3);
        float tanh_s = std::tanh(s);
        float sech_s = 1.0f / std::cosh(s);
        float sech2_s = sech_s * sech_s;

        float term1 = 0.5f * (1.0f + tanh_s);
        float term2 = 0.5f * x * sech2_s * 0.7978845608f * (1.0f + 3.0f * 0.044715f * x2);
        return term1 + term2;
    });

    return dout * d_gelu;
}
