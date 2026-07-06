#ifndef LAYERNORMLAYER_H
#define LAYERNORMLAYER_H

#include "Layer.h"
#include <cmath>
#include <iostream>

class LayerNormLayer : public Layer {
public: // Made public for testing and optimizer parameter updates
    int channels;
    float eps;

    Tensor scale; // Gamma parameter vector of shape {channels}
    Tensor shift; // Beta parameter vector of shape {channels}

private:
    // Cached states for backward pass
    Tensor cached_input;
    Tensor cached_mean;
    Tensor cached_var;
    Tensor cached_x_hat;

public:
    explicit LayerNormLayer(int channels, float eps = 1e-5f);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& dout) override;
    std::vector<Tensor*> get_parameters() override;
};

#endif //LAYERNORMLAYER_H
