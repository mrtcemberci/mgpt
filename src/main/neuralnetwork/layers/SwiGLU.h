#ifndef SWIGLU_H
#define SWIGLU_H

#include "Layer.h"
#include "LinearLayer.h"
#include "Scratchpad.h"

class SwiGLU : public Layer {
private:
    LinearLayer mlp_gate;
    LinearLayer mlp_up;
    LinearLayer mlp_down;

    // Cached tensors for the backward pass
    Tensor cached_gate_out;
    Tensor cached_up_out;
    Tensor cached_swiglu_tmp;
    Tensor cached_out;
    Tensor cached_dX;

    Tensor cached_d_down;
    Tensor cached_d_gate;
    Tensor cached_d_up;
    Tensor cached_d_gate_proj;
    Tensor cached_d_up_proj;

public:
    SwiGLU(int embed_dim, int hidden_dim = -1);

    void set_scratchpad(Scratchpad* pad) override;

    Tensor forward(const Tensor& input) override;
    void forward_into(const Tensor& input, Tensor& output) override;

    Tensor backward(const Tensor& dout) override;
    void backward_into(const Tensor& dout, Tensor& din) override;

    std::vector<Tensor*> get_parameters() override;
    ScratchpadFootprint get_footprint(int B, int T) override;
};

#endif // SWIGLU_H
