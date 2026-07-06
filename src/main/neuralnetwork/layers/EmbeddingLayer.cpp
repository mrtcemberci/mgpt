#include "EmbeddingLayer.h"

EmbeddingLayer::EmbeddingLayer(int table_size, int embed_dim)
    : table_size(table_size),
      embed_dim(embed_dim),
      lookup_table(Tensor::randn({table_size, embed_dim}, 0.0f, 0.02f)) {
}

// FORWARD PASS: Look up row index for each integer element in input
Tensor EmbeddingLayer::forward(const Tensor& input) {
    cached_input = input;
    return lookup_table.embedding_lookup(input);
}

// BACKWARD PASS: Accumulate downstream gradients into lookup table rows
Tensor EmbeddingLayer::backward(const Tensor& dout) {
    lookup_table.embedding_backward(dout, cached_input);
    return Tensor();
}

std::vector<Tensor*> EmbeddingLayer::get_parameters() {
    return { &lookup_table };
}
