#ifndef TRANSFORMERBLOCK_H
#define TRANSFORMERBLOCK_H

#include "Layer.h"
#include "LayerNormLayer.h"
#include "MultiHeadAttentionLayer.h"
#include "LinearLayer.h"
#include "GELULayer.h"
#include <vector>

class TransformerBlock : public Layer {
public: // Public for optimizer updates and inspection
    int channels;
    LayerNormLayer ln1;               // Pre-attention layer normalization
    MultiHeadAttentionLayer attn;     // Causal multi-head self-attention
    LayerNormLayer ln2;               // Pre-MLP layer normalization
    LinearLayer mlp_fc1;              // Feed-forward expansion: channels -> 4 * channels
    GELULayer act;                    // GELU activation function
    LinearLayer mlp_fc2;              // Feed-forward contraction: 4 * channels -> channels

    Tensor cached_input;              // Publicly accessible for gradient checkpointing recomputation

private: // Private cached states for backpropagation
    Tensor cached_ln1_out;
    Tensor cached_attn_out;
    Tensor cached_x1; // Intermediate state: input + attn(ln1(input))
    Tensor cached_ln2_out;
    Tensor cached_fc1_out;
    Tensor cached_act_out;
    Tensor cached_fc2_out;
    Tensor cached_d_fc2;
    Tensor cached_d_act;
    Tensor cached_d_fc1;
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
        mlp_fc1.set_scratchpad(pad);
        act.set_scratchpad(pad);
        mlp_fc2.set_scratchpad(pad);
        ln2.set_scratchpad(pad);
    }
};

#endif //TRANSFORMERBLOCK_H
