#ifndef LAYER_H
#define LAYER_H
#include "Tensor.h"



struct ScratchpadFootprint {
    size_t fwd_standing;
    size_t fwd_temp;
    size_t bwd_temp;

    size_t fwd_peak() const { return fwd_standing + fwd_temp; }
    size_t bwd_peak() const { return fwd_standing + bwd_temp; }
};

class Scratchpad;


class Layer {
protected:
    Scratchpad* scratchpad = nullptr;
public:
    virtual ~Layer() = default;

    virtual void set_scratchpad(Scratchpad* pad) { scratchpad = pad; }

    // Execute mathematical transformation and cache necessary states
    virtual Tensor forward(const Tensor& input) = 0;
    virtual void forward_into(const Tensor& input, Tensor& output) { output = forward(input); }

    // Receive downstream gradient (dout), compute internal weight/bias gradients, return upstream gradient (din)
    virtual Tensor backward(const Tensor& dout) = 0;
    virtual void backward_into(const Tensor& dout, Tensor& din) { din = backward(dout); }

    // Return pointers to all learnable weight/bias Tensors within this layer (used by Optimizers)
    virtual std::vector<Tensor*> get_parameters() { return {}; }
    virtual ScratchpadFootprint get_footprint(int B, int T) { return {0, 0, 0}; }
    
    // Clear heavy cached activations to save memory when gradient checkpointing is enabled
    virtual void clear_activations() {}

    // Migrate all learnable parameters in this layer to the specified device
    virtual void to(Device target_device) {
        for (Tensor* param : get_parameters()) {
            if (param) param->to(target_device);
        }
    }
};


#endif //LAYER_H
