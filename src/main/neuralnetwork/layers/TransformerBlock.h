#ifndef TRANSFORMERBLOCK_H
#define TRANSFORMERBLOCK_H

#include "Layer.h"
#include "LayerNormLayer.h"
#include "RMSNormLayer.h"
#include "MultiHeadAttentionLayer.h"
#include "MoELayer.h"
#include <vector>

class TransformerBlock : public Layer {
public: // Public for optimizer updates and inspection
    int channels;
    RMSNormLayer ln1;                 // Pre-attention RMS normalization
    MultiHeadAttentionLayer attn;     // Causal multi-head self-attention
    RMSNormLayer ln2;                 // Pre-MLP RMS normalization
    MoELayer mlp;                     // Mixture of Experts MLP block

    Tensor cached_input;              // Publicly accessible for gradient checkpointing recomputation

private: // Private cached states for backpropagation
    Tensor cached_ln1_out;
    Tensor cached_attn_out;
    Tensor cached_x1; // Intermediate state: input + attn(ln1(input))
    Tensor cached_ln2_out;
    Tensor cached_d_ln2;
    Tensor cached_d_x1;
    Tensor cached_d_attn;
    Tensor cached_d_ln1;
    Tensor cached_dX;

public:
    Tensor cached_out;
    explicit TransformerBlock(int channels, int num_heads, int num_experts, int top_k, bool use_flash_attention);

    Tensor forward(const Tensor& input) override;
    void forward_into(const Tensor& input, Tensor& output) override;
    Tensor backward(const Tensor& dout) override;
    void backward_into(const Tensor& dout, Tensor& din) override;
    std::vector<Tensor*> get_parameters() override;
    ScratchpadFootprint get_footprint(int B, int T) override;

    void set_scratchpad(Scratchpad* pad) override {
        this->scratchpad = pad;
        attn.set_scratchpad(pad);
        ln1.set_scratchpad(pad);
        mlp.set_scratchpad(pad);
        ln2.set_scratchpad(pad);
    }
};

#endif //TRANSFORMERBLOCK_H
