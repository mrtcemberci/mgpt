#ifndef LAYER_H
#define LAYER_H
#include "Tensor.h"


class Layer {
public:
    virtual ~Layer() = default;

    // Execute mathematical transformation and cache necessary states
    virtual Tensor forward(const Tensor& input) = 0;

    // Receive downstream gradient (dout), compute internal weight/bias gradients, return upstream gradient (din)
    virtual Tensor backward(const Tensor& dout) = 0;

    // Return pointers to all learnable weight/bias Tensors within this layer (used by Optimizers)
    virtual std::vector<Tensor*> get_parameters() { return {}; }
};


#endif //LAYER_H
