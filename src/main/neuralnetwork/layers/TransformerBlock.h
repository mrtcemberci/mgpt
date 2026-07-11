#ifndef TRANSFORMERBLOCK_H
#define TRANSFORMERBLOCK_H

#include "Layer.h"
#include "LayerNormLayer.h"
#include "RMSNormLayer.h"
#include "MultiHeadAttentionLayer.h"
#include "LinearLayer.h"
#include <vector>

class TransformerBlock : public Layer {
public: // Public for optimizer updates and inspection
    int channels;
    RMSNormLayer ln1;                 // Pre-attention RMS normalization
    MultiHeadAttentionLayer attn;     // Causal multi-head self-attention
    RMSNormLayer ln2;                 // Pre-MLP RMS normalization
    LinearLayer mlp_gate;             // SwiGLU gate projection: channels -> 4 * channels
    LinearLayer mlp_up;               // SwiGLU up projection:   channels -> 4 * channels
    LinearLayer mlp_down;             // SwiGLU down projection: 4 * channels -> channels

    Tensor cached_input;              // Publicly accessible for gradient checkpointing recomputation

private: // Private cached states for backpropagation
    Tensor cached_ln1_out;
    Tensor cached_attn_out;
    Tensor cached_x1; // Intermediate state: input + attn(ln1(input))
    Tensor cached_ln2_out;
    Tensor cached_gate_out;           // Pre-swish gate projection output (needed for swish backward)
    Tensor cached_gate_swished;       // Post-swish: swish(gate)            (needed for d_up)
    Tensor cached_up_out;             // Up projection output               (needed for d_gate_swish)
    Tensor cached_swiglu_tmp;         // CPU-only fallback for swish(gate) ⊙ up (GPU uses scratchpad)
    Tensor cached_d_down;
    Tensor cached_d_gate;
    Tensor cached_d_up;
    Tensor cached_d_swish;
    Tensor cached_d_gate_proj;
    Tensor cached_d_up_proj;
    Tensor cached_d_ln2;
    Tensor cached_d_x1;
    Tensor cached_d_attn;
    Tensor cached_d_ln1;
    Tensor cached_dX;

public:
    Tensor cached_out;
    explicit TransformerBlock(int channels, int num_heads = 6);

    Tensor forward(const Tensor& input) override;
    void forward_into(const Tensor& input, Tensor& output) override;
    Tensor backward(const Tensor& dout) override;
    void backward_into(const Tensor& dout, Tensor& din) override;
    std::vector<Tensor*> get_parameters() override;

    void set_scratchpad(Scratchpad* pad) override {
        this->scratchpad = pad;
        attn.set_scratchpad(pad);
        ln1.set_scratchpad(pad);
        mlp_gate.set_scratchpad(pad);
        mlp_up.set_scratchpad(pad);
        mlp_down.set_scratchpad(pad);
        ln2.set_scratchpad(pad);
    }
};

#endif //TRANSFORMERBLOCK_H
