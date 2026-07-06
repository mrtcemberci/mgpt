#include "EmbeddingLayer.h"

EmbeddingLayer::EmbeddingLayer(int table_size, int embed_dim)
    : table_size(table_size),
      embed_dim(embed_dim),
      lookup_table({table_size, embed_dim}, 0.01f) {
}

// FORWARD PASS: Look up row index for each integer element in input
Tensor EmbeddingLayer::forward(const Tensor& input) {
    cached_input = input;

    std::vector<int> out_shape = input.shape;
    out_shape.push_back(embed_dim);
    Tensor output(out_shape, 0.0f);

    int total_tokens = (int)input.size();

    for (int i = 0; i < total_tokens; ++i) {
        int token_id = (int)input.data[i];

        if (token_id < 0 || token_id >= table_size) {
            std::cerr << "EmbeddingLayer::forward: token_id " << token_id 
                      << " is out of bounds (table_size: " << table_size << ")!" << std::endl;
            exit(-1);
        }

        const float* table_row = lookup_table.data.data() + (token_id * embed_dim);
        float* out_row = output.data.data() + (i * embed_dim);

        for (int j = 0; j < embed_dim; ++j) {
            out_row[j] = table_row[j];
        }
    }

    return output;
}

// BACKWARD PASS: Accumulate downstream gradients into lookup table rows
Tensor EmbeddingLayer::backward(const Tensor& dout) {
    int total_tokens = (int)cached_input.size();

    for (int i = 0; i < total_tokens; ++i) {
        int token_id = (int)cached_input.data[i];

        if (token_id < 0 || token_id >= table_size) {
            std::cerr << "EmbeddingLayer::backward: token_id " << token_id 
                      << " is out of bounds!" << std::endl;
            exit(-1);
        }

        const float* dout_row = dout.data.data() + (i * embed_dim);
        float* grad_row = lookup_table.grad.data() + (token_id * embed_dim);

        for (int j = 0; j < embed_dim; ++j) {
            grad_row[j] += dout_row[j];
        }
    }

    return Tensor();
}

std::vector<Tensor*> EmbeddingLayer::get_parameters() {
    return { &lookup_table };
}
