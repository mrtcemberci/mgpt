#ifndef GELULAYER_H
#define GELULAYER_H

#include "Layer.h"
#include <cmath>

class GELULayer : public Layer {
private:
    Tensor cached_input;
public:
    GELULayer() = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& dout) override;
    std::vector<Tensor*> get_parameters() override { return {}; } // GELU has no learnable weights/biases
};

#endif //GELULAYER_H
