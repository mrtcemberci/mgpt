#ifndef GPT_H
#define GPT_H

#include "Layer.h"
#include "EmbeddingLayer.h"
#include "TransformerBlock.h"
#include "LayerNormLayer.h"
#include "LinearLayer.h"
#include "CrossEntropyLossLayer.h"
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>

struct GPTConfig {
    int vocab_size = 65;   // ~65 unique ASCII characters in Shakespeare text
    int max_seq_len = 256; // Context window size T = 256 (allows attending across lines and dialogue turns)
    int embed_dim = 384;   // Channel dimension C = 384 (high representational capacity for character grammar)
    int num_layers = 6;    // 6 sequential Transformer blocks (hierarchical feature learning)
    int num_heads = 6;     // 6 parallel attention heads (each head size = 384 / 6 = 64)
};

class GPT : public Layer {
public: // Public for inspection and optimizer updates
    GPTConfig config;
    EmbeddingLayer tok_emb;                                    // Token embeddings: V x C
    EmbeddingLayer pos_emb;                                    // Positional embeddings: T x C
    std::vector<std::unique_ptr<TransformerBlock>> blocks;     // Stack of L Transformer blocks
    LayerNormLayer ln_f;                                       // Final layer normalization
    LinearLayer lm_head;                                       // Language model projection: C x V
    CrossEntropyLossLayer loss_layer;                          // Softmax & Cross-Entropy loss calculation

private: // Private cached states for backpropagation
    Tensor cached_input_ids;
    Tensor cached_tok_emb;
    Tensor cached_pos_emb;
    Tensor cached_pos_ids;
    Tensor cached_x0;
    Tensor cached_ln_f_out;
    Tensor cached_logits;
    Tensor cached_d_ln_f;
    Tensor cached_d_curr;
    std::vector<Tensor> cached_d_blocks;
    Tensor cached_d_logits;
    Tensor cached_dX;
    Tensor cached_dummy_din;

public:
    explicit GPT(const GPTConfig& config);

    // Forward pass: Takes integer token IDs tensor of shape {Batch, Time}
    // Returns unnormalized vocabulary logits of shape {Batch, Time, VocabSize}
    Tensor forward(const Tensor& input_ids) override;
    void forward_into(const Tensor& input_ids, Tensor& output) override;

    // Evaluates loss against target IDs {Batch, Time} and initiates backprop
    float compute_loss(const Tensor& logits, const Tensor& target_ids);
    Tensor backward(const Tensor& d_logits) override;
    void backward_into(const Tensor& d_logits, Tensor& din) override;

    std::vector<Tensor*> get_parameters() override;

    // Export & Import weights to raw binary .bin file for lightweight inference
    void save_weights_bin(const std::string& filepath);
    void load_weights_bin(const std::string& filepath);
};

#endif //GPT_H
