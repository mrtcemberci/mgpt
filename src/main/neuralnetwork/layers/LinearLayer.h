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
private:
    Tensor cached_input;
    Tensor cached_output;
    Tensor cached_sum_rows;
    Tensor cached_input_T;
    Tensor cached_dW;
    Tensor cached_W_T;
    Tensor cached_dX;

public:
    LinearLayer(int in_channels, int out_channels);

    Tensor forward(const Tensor& input) override;
    void forward_into(const Tensor& input, Tensor& output) override;
    Tensor backward(const Tensor& dout) override;
    void backward_into(const Tensor& dout, Tensor& din) override;
    std::vector<Tensor*> get_parameters() override;
};


#endif //LINEARLAYER_H
