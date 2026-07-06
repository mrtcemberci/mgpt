#ifndef EMBEDDINGLAYER_H
#define EMBEDDINGLAYER_H

#include "Layer.h"
#include <iostream>
#include <vector>

class EmbeddingLayer : public Layer {
public: // Made public for optimizer updates and testing
    int table_size;
    int embed_dim;
    Tensor lookup_table;

private:
    Tensor cached_input;

public:
    EmbeddingLayer(int table_size, int embed_dim);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& dout) override;
    std::vector<Tensor*> get_parameters() override;
};

#endif //EMBEDDINGLAYER_H
