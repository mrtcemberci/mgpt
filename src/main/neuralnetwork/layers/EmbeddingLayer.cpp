#include "EmbeddingLayer.h"

/**
 * Takes in the BxT tensor and produces BxTxC of token embeddings, does not have positional embeddings
 * 
 * Lookup table is VxC (lookup array of V indices to produce a C length array)
 */

EmbeddingLayer::EmbeddingLayer(int table_size, int embed_dim)
    : table_size(table_size),
      embed_dim(embed_dim),
      lookup_table(Tensor::randn({table_size, embed_dim}, 0.0f, 0.02f)) {
}

// FORWARD PASS: Look up row index for each integer element in input
void EmbeddingLayer::forward_into(const Tensor& input, Tensor& output) {
    cached_input = input;
    lookup_table.embedding_lookup_into(input, output);
    cached_output = output;
}

Tensor EmbeddingLayer::forward(const Tensor& input) {
    forward_into(input, cached_output);
    return cached_output;
}

// BACKWARD PASS: Accumulate downstream gradients into lookup table rows
void EmbeddingLayer::backward_into(const Tensor& dout, Tensor& din) {
    lookup_table.embedding_backward(dout, cached_input);
}

Tensor EmbeddingLayer::backward(const Tensor& dout) {
    backward_into(dout, cached_output);
    return Tensor();
}

std::vector<Tensor*> EmbeddingLayer::get_parameters() {
    return { &lookup_table };
}

// does not use the scratch pad
ScratchpadFootprint EmbeddingLayer::get_footprint(int B, int T) { return {0, 0, 0}; }
