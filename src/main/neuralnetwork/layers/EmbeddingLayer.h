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
    Tensor cached_output;

public:
    EmbeddingLayer(int table_size, int embed_dim);

    Tensor forward(const Tensor& input) override;
    void forward_into(const Tensor& input, Tensor& output) override;
    Tensor backward(const Tensor& dout) override;
    void backward_into(const Tensor& dout, Tensor& din) override;
    std::vector<Tensor*> get_parameters() override;
    ScratchpadFootprint get_footprint(int B, int T) override;
};

#endif //EMBEDDINGLAYER_H
