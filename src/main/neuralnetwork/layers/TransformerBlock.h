#ifndef TRANSFORMERBLOCK_H
#define TRANSFORMERBLOCK_H

#include "Layer.h"
#include "LayerNormLayer.h"
#include "SingleHeadAttentionLayer.h"
#include "LinearLayer.h"
#include "GELULayer.h"
#include <vector>

class TransformerBlock : public Layer {
public: // Public for optimizer updates and inspection
    int channels;
    LayerNormLayer ln1;               // Pre-attention layer normalization
    SingleHeadAttentionLayer attn;    // Causal self-attention
    LayerNormLayer ln2;               // Pre-MLP layer normalization
    LinearLayer mlp_fc1;              // Feed-forward expansion: channels -> 4 * channels
    GELULayer act;                    // GELU activation function
    LinearLayer mlp_fc2;              // Feed-forward contraction: 4 * channels -> channels

private: // Private cached states for backpropagation
    Tensor cached_input;
    Tensor cached_ln1_out;
    Tensor cached_attn_out;
    Tensor cached_x1; // Intermediate state: input + attn(ln1(input))
    Tensor cached_ln2_out;
    Tensor cached_fc1_out;
    Tensor cached_act_out;

public:
    explicit TransformerBlock(int channels);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& dout) override;
    std::vector<Tensor*> get_parameters() override;
};

#endif //TRANSFORMERBLOCK_H
