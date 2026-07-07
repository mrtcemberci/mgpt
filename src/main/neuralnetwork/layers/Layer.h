#ifndef LAYER_H
#define LAYER_H
#include "Tensor.h"


class Layer {
public:
    virtual ~Layer() = default;

    // Execute mathematical transformation and cache necessary states
    virtual Tensor forward(const Tensor& input) = 0;
    virtual void forward_into(const Tensor& input, Tensor& output) { output = forward(input); }

    // Receive downstream gradient (dout), compute internal weight/bias gradients, return upstream gradient (din)
    virtual Tensor backward(const Tensor& dout) = 0;
    virtual void backward_into(const Tensor& dout, Tensor& din) { din = backward(dout); }

    // Return pointers to all learnable weight/bias Tensors within this layer (used by Optimizers)
    virtual std::vector<Tensor*> get_parameters() { return {}; }

    // Migrate all learnable parameters in this layer to the specified device
    virtual void to(Device target_device) {
        for (Tensor* param : get_parameters()) {
            if (param) param->to(target_device);
        }
    }
};


#endif //LAYER_H
