//
// Created by Mertd on 7/6/2026.
//

#ifndef LINEARLAYER_H
#define LINEARLAYER_H

#include "Layer.h"
#include <vector>

class LinearLayer : public Layer {
public:
    int in_channels;
    int out_channels;
    Tensor weights;
    Tensor biases;
    Tensor cached_input;

public:
    LinearLayer(int in_channels, int out_channels);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& dout) override;
    std::vector<Tensor*> get_parameters() override;
};


#endif //LINEARLAYER_H
