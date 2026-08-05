#ifndef GELULAYER_H
#define GELULAYER_H

#include "Layer.h"
#include <cmath>

class GELULayer : public Layer {
private:
    Tensor cached_input;
    Tensor cached_output;
    Tensor cached_d_gelu_workspace;
    Tensor cached_dX;
public:
    GELULayer() = default;

    Tensor forward(const Tensor& input) override;
    void forward_into(const Tensor& input, Tensor& output) override;
    Tensor backward(const Tensor& dout) override;
    void backward_into(const Tensor& dout, Tensor& din) override;
    std::vector<Tensor*> get_parameters() override { return {}; } // GELU has no learnable weights/biases
    ScratchpadFootprint get_footprint(int B, int T) override;
};

#endif //GELULAYER_H
