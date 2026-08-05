#include "MoELayer.h"
#include "Scratchpad.h"

MoELayer::MoELayer(int embed_dim, int hidden_dim, int num_experts, int top_k)
    : num_experts(num_experts), 
      top_k(top_k), 
      router(embed_dim, num_experts) 
{
    for (int i = 0; i < num_experts; i++) {
        experts.emplace_back(embed_dim, hidden_dim);
    }
}

void MoELayer::set_scratchpad(Scratchpad* pad) {
    Layer::set_scratchpad(pad);
    router.set_scratchpad(pad);
    
    for (auto& expert : experts) {
        expert.set_scratchpad(pad);
    }
}

Tensor MoELayer::forward(const Tensor& input) {
    forward_into(input, cached_out);
    return cached_out;
}

void MoELayer::forward_into(const Tensor& input, Tensor& output) {

    size_t B = input.shape[0], T = input.shape[1], C = input.shape[2];
    
    if (scratchpad && input.device == Device::CUDA) {
        cached_router_logits = Tensor::view({(int)B, (int)T, (int)num_experts}, scratchpad->get_address(B*T*num_experts), Device::CUDA);
        cached_router_probs = Tensor::view({(int)B, (int)T, (int)num_experts}, scratchpad->get_address(B*T*num_experts), Device::CUDA);
        cached_top_indices = Tensor::view({(int)B, (int)T, top_k}, scratchpad->get_address(B*T*top_k), Device::CUDA);
        cached_top_probs = Tensor::view({(int)B, (int)T, top_k}, scratchpad->get_address(B*T*top_k), Device::CUDA);
        
        cached_gathered_tokens = Tensor::view({(int)B * (int)T * top_k, (int)C}, scratchpad->get_address(B*T*top_k*C), Device::CUDA);
        cached_sorted_token_ids = Tensor::view({(int)B * (int)T * top_k}, scratchpad->get_address(B*T*top_k), Device::CUDA);
        
        cached_out = Tensor::view({(int)B, (int)T, (int)C}, scratchpad->get_address(B*T*C), Device::CUDA);
    } else {
        if (cached_router_logits.shape.empty()) cached_router_logits = Tensor({(int)B, (int)T, (int)num_experts}, 0.0f, input.device);
        if (cached_router_probs.shape.empty()) cached_router_probs = Tensor({(int)B, (int)T, (int)num_experts}, 0.0f, input.device);
        if (cached_top_indices.shape.empty()) cached_top_indices = Tensor({(int)B, (int)T, top_k}, 0.0f, input.device);
        if (cached_top_probs.shape.empty()) cached_top_probs = Tensor({(int)B, (int)T, top_k}, 0.0f, input.device);
        if (cached_gathered_tokens.shape.empty()) cached_gathered_tokens = Tensor({(int)B * (int)T * top_k, (int)C}, 0.0f, input.device);
        if (cached_sorted_token_ids.shape.empty()) cached_sorted_token_ids = Tensor({(int)B * (int)T * top_k}, 0.0f, input.device);
        if (cached_out.shape.empty()) cached_out = Tensor({(int)B, (int)T, (int)C}, 0.0f, input.device);
    }

    router.forward_into(input, cached_router_logits);

    // logits of BxTxExperts
    cached_router_logits.softmax_into(-1, cached_router_probs);
    
    // select top k form tensor of BxTxK
    cached_router_probs.top_k_into(top_k, -1, cached_top_indices, cached_top_probs);
    
    // form tensor of num_experts containing count of each expert
    if (cached_expert_counts.shape.empty() || cached_expert_counts.device != input.device) {
        cached_expert_counts = Tensor({num_experts}, 0.0f, input.device);
    }
    cached_expert_counts.fill(0.0f);
    cached_top_indices.moe_histogram_into(num_experts, cached_expert_counts);
    
    cached_cpu_counts.resize(num_experts);
    cached_expert_counts.copy_to_host(cached_cpu_counts.data());
    
    cached_cpu_offsets.resize(num_experts);
    float current_offset = 0;
    for (int i = 0; i < num_experts; i++) {
        cached_cpu_offsets[i] = current_offset;
        current_offset += cached_cpu_counts[i];
    }
    
    if (cached_expert_offsets.shape.empty() || cached_expert_offsets.device != input.device) {
        cached_expert_offsets = Tensor({num_experts}, 0.0f, input.device);
    }
    cached_expert_offsets.copy_from_host(cached_cpu_offsets.data());
    
    // Gather tokens into experts, populate tensor of gathered tokens and a map of current token to old spot
    cached_top_indices.moe_gather_into(input, cached_expert_offsets, cached_gathered_tokens, cached_sorted_token_ids);
    
    cached_fwd_savepoint = 0;
    if (scratchpad && input.device == Device::CUDA) {
        cached_fwd_savepoint = scratchpad->get_savepoint();
    }
    
    cached_gathered_outputs = (scratchpad && input.device == Device::CUDA)
        ? Tensor::view({(int)B * (int)T * top_k, (int)C}, scratchpad->get_address(B*T*top_k*C), Device::CUDA)
        : Tensor({(int)B * (int)T * top_k, (int)C}, 0.0f, input.device);
        
    for (int i = 0; i < num_experts; i++) {
        int count = (int)cached_cpu_counts[i];
        if (count == 0) continue;
        
        int offset = (int)cached_cpu_offsets[i];
        
        Tensor expert_input;
        Tensor expert_output;
        // splice the expert inputs and pass their expert output result location
        if (input.device == Device::CUDA) {
            expert_input = Tensor::view({1, count, (int)C}, cached_gathered_tokens.get_data_ptr() + offset * C, Device::CUDA);
            expert_output = Tensor::view({1, count, (int)C}, cached_gathered_outputs.get_data_ptr() + offset * C, Device::CUDA);
        } else {
            expert_input = Tensor({1, count, (int)C}, 0.0f, Device::CPU);
            expert_output = Tensor({1, count, (int)C}, 0.0f, Device::CPU);
            std::memcpy((void*)expert_input.get_data_ptr(), cached_gathered_tokens.get_data_ptr() + offset * C, count * C * sizeof(float));
        }
        
        experts[i].forward_into(expert_input, expert_output);
        
        // copy the output if on CPU
        if (input.device != Device::CUDA) {
            std::memcpy((void*)(cached_gathered_outputs.get_data_ptr() + offset * C), expert_output.get_data_ptr(), count * C * sizeof(float));
        }
    }
    
    // return gathered values to original spots
    if (output.shape.empty()) output = Tensor({(int)B, (int)T, (int)C}, 0.0f, input.device);
    output.fill(0.0f);
    Tensor::moe_scatter_into(cached_gathered_outputs, cached_sorted_token_ids, cached_top_probs, output);
    
    cached_out = output;
    
    // Do NOT restore savepoint here because SwiGLU experts need their cached activations alive for backward pass
}

Tensor MoELayer::backward(const Tensor& dout) {
    backward_into(dout, cached_dX);
    return cached_dX;
}

void MoELayer::backward_into(const Tensor& dout, Tensor& din) {
    size_t B = dout.shape[0], T = dout.shape[1], C = dout.shape[2];
    
    // We need dGathered_outputs and dGathered_inputs buffers
    size_t bwd_savepoint = 0;
    if (scratchpad && dout.device == Device::CUDA) {
        bwd_savepoint = scratchpad->get_savepoint();
    }
    
    Tensor dGathered_outputs = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({(int)B * (int)T * top_k, (int)C}, scratchpad->get_address(B*T*top_k*C), Device::CUDA)
        : Tensor({(int)B * (int)T * top_k, (int)C}, 0.0f, dout.device);
        
    Tensor dGathered_inputs = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({(int)B * (int)T * top_k, (int)C}, scratchpad->get_address(B*T*top_k*C), Device::CUDA)
        : Tensor({(int)B * (int)T * top_k, (int)C}, 0.0f, dout.device);
        
    Tensor d_top_probs = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({(int)B, (int)T, top_k}, scratchpad->get_address(B*T*top_k), Device::CUDA)
        : Tensor({(int)B, (int)T, top_k}, 0.0f, dout.device);
        
    Tensor::moe_scatter_backward_into(dout, cached_gathered_outputs, cached_sorted_token_ids, cached_top_probs, dGathered_outputs, d_top_probs);
    
    // Expert Backward
    for (int i = 0; i < num_experts; i++) {
        int count = (int)cached_cpu_counts[i];
        if (count == 0) continue;
        
        int offset = (int)cached_cpu_offsets[i];
        
        Tensor expert_dout;
        Tensor expert_din;
        if (dout.device == Device::CUDA) {
            expert_dout = Tensor::view({1, count, (int)C}, dGathered_outputs.get_data_ptr() + offset * C, Device::CUDA);
            expert_din = Tensor::view({1, count, (int)C}, dGathered_inputs.get_data_ptr() + offset * C, Device::CUDA);
        } else {
            expert_dout = Tensor({1, count, (int)C}, 0.0f, Device::CPU);
            expert_din = Tensor({1, count, (int)C}, 0.0f, Device::CPU);
            std::memcpy((void*)expert_dout.get_data_ptr(), dGathered_outputs.get_data_ptr() + offset * C, count * C * sizeof(float));
        }
        
        experts[i].backward_into(expert_dout, expert_din);
        
        if (dout.device != Device::CUDA) {
            std::memcpy((void*)(dGathered_inputs.get_data_ptr() + offset * C), expert_din.get_data_ptr(), count * C * sizeof(float));
        }
    }
    
    // Gather Backward
    if (din.shape.empty()) din = Tensor({(int)B, (int)T, (int)C}, 0.0f, dout.device);
    din.fill(0.0f);
    Tensor::moe_gather_backward_into(dGathered_inputs, cached_sorted_token_ids, din, top_k);
    
    // Scatter top-k task gradients into full 8-expert probability array
    Tensor d_router_probs = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({(int)B, (int)T, num_experts}, scratchpad->get_address(B*T*num_experts), Device::CUDA)
        : Tensor({(int)B, (int)T, num_experts}, 0.0f, dout.device);
    Tensor::scatter_indices_into(d_top_probs, cached_top_indices, d_router_probs, num_experts, top_k);
    
    float alpha = 0.01f;
    std::vector<float> lb_grads(num_experts, 0.0f);
    for (int i = 0; i < num_experts; i++) {
        float f_i = cached_cpu_counts[i] / (float)(B * T);
        lb_grads[i] = (alpha * num_experts * f_i) / (float)(B * T);
    }
    Tensor lb_grads_tensor = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({1, num_experts}, scratchpad->get_address(num_experts), Device::CUDA)
        : Tensor({1, num_experts}, 0.0f, dout.device);
    lb_grads_tensor.copy_from_host(lb_grads.data());
    d_router_probs.add_broadcast_in_place(lb_grads_tensor);
    
    Tensor d_router_logits = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({(int)B, (int)T, num_experts}, scratchpad->get_address(B*T*num_experts), Device::CUDA)
        : Tensor({(int)B, (int)T, num_experts}, 0.0f, dout.device);
        
    cached_router_probs.softmax_backward_into(d_router_probs, d_router_logits);
    
    Tensor router_din = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({(int)B, (int)T, (int)C}, scratchpad->get_address(B*T*C), Device::CUDA)
        : Tensor({(int)B, (int)T, (int)C}, 0.0f, dout.device);
    router.backward_into(d_router_logits, router_din);
    
    Tensor::add_into(din, router_din, din);
    cached_dX = din;
    
    if (scratchpad && dout.device == Device::CUDA) {
        scratchpad->restore_savepoint(bwd_savepoint);
    }
}

std::vector<Tensor*> MoELayer::get_parameters() {
    std::vector<Tensor*> params;
    auto p_router = router.get_parameters();
    params.insert(params.end(), p_router.begin(), p_router.end());
    
    for (auto& expert : experts) {
        auto p_exp = expert.get_parameters();
        params.insert(params.end(), p_exp.begin(), p_exp.end());
    }
    
    return params;
}

ScratchpadFootprint MoELayer::get_footprint(int B, int T) {
    size_t C = router.in_channels;
    // Add 12 * B * T * top_k * C for SwiGLU expert activations left on scratchpad
    size_t fwd_standing = (size_t)(B * T * num_experts * 2 + B * T * top_k * 2 + B * T * top_k * C + B * T * top_k + B * T * C) + (size_t)(12 * B * T * top_k * C);
    size_t bwd_standing = (size_t)(B * T * top_k * C * 2 + B * T * num_experts * 2 + B * T * top_k);
    size_t max_expert_fwd = 0;
    size_t max_expert_bwd = 0;
    for (auto& expert : experts) {
        auto fp = expert.get_footprint(B * top_k, T);
        if (fp.fwd_peak() > max_expert_fwd) max_expert_fwd = fp.fwd_peak();
        if (fp.bwd_peak() > max_expert_bwd) max_expert_bwd = fp.bwd_peak();
    }
    return {fwd_standing, max_expert_fwd, bwd_standing + max_expert_bwd};
}
