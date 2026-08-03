#ifndef MOELAYER_H
#define MOELAYER_H

#include "Layer.h"
#include "LinearLayer.h"
#include "SwiGLU.h"
#include <vector>

class MoELayer : public Layer {
private:
    int num_experts;
    int top_k;

    LinearLayer router;
    std::vector<SwiGLU> experts;

    // Cached tensors for backward pass
    Tensor cached_router_logits;
    Tensor cached_router_probs;
    Tensor cached_out;
    Tensor cached_dX;
    Tensor cached_top_indices;
    Tensor cached_top_probs;
    Tensor cached_gathered_tokens;
    Tensor cached_sorted_token_ids;
    
    // Cached for backward pass
    std::vector<float> cached_cpu_counts;
    std::vector<float> cached_cpu_offsets;
    // std::vector<std::vector<int>> cached_expert_indices; 

public:
    MoELayer(int embed_dim, int hidden_dim, int num_experts, int top_k);

    void set_scratchpad(Scratchpad* pad) override;

    Tensor forward(const Tensor& input) override;
    void forward_into(const Tensor& input, Tensor& output) override;

    Tensor backward(const Tensor& dout) override;
    void backward_into(const Tensor& dout, Tensor& din) override;

    std::vector<Tensor*> get_parameters() override;
};

#endif // MOELAYER_H
