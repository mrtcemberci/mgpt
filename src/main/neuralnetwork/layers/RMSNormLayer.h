#ifndef RMSNORMLAYER_H
#define RMSNORMLAYER_H

#include "Layer.h"
#include <cmath>
#include <iostream>

class RMSNormLayer : public Layer {
public:
    int channels;
    float eps;

    Tensor scale; // Gamma parameter vector of shape {channels}

private:
    Tensor cached_input;
    Tensor cached_rsqrt;
    Tensor cached_x_hat;
    Tensor cached_output;
    Tensor cached_dX;

public:
    explicit RMSNormLayer(int channels, float eps = 1e-5f);
    Tensor forward(const Tensor& input) override;
    void forward_into(const Tensor& input, Tensor& output) override;
    Tensor backward(const Tensor& dout) override;
    void backward_into(const Tensor& dout, Tensor& din) override;
    std::vector<Tensor*> get_parameters() override;
    ScratchpadFootprint get_footprint(int B, int T) override;
};

#endif //RMSNORMLAYER_H
