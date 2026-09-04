#include "SwiGLU.h"
#include "Scratchpad.h"

SwiGLU::SwiGLU(int embed_dim, int hidden_dim) 
    : mlp_gate(embed_dim, hidden_dim == -1 ? embed_dim * 4 : hidden_dim),
      mlp_up(embed_dim, hidden_dim == -1 ? embed_dim * 4 : hidden_dim),
      mlp_down(hidden_dim == -1 ? embed_dim * 4 : hidden_dim, embed_dim) 
{
}

void SwiGLU::set_scratchpad(Scratchpad* pad) {
    Layer::set_scratchpad(pad);
    mlp_gate.set_scratchpad(pad);
    mlp_up.set_scratchpad(pad);
    mlp_down.set_scratchpad(pad);
}

Tensor SwiGLU::forward(const Tensor& input) {
    forward_into(input, cached_out);
    return cached_out;
}

void SwiGLU::forward_into(const Tensor& input, Tensor& output) {
    if (scratchpad && input.device == Device::CUDA) {
        size_t B = input.shape[0], T = input.shape[1], C = input.shape[2];
        cached_gate_out = Tensor::view({(int)B, (int)T, (int)C*4}, scratchpad->get_address(B*T*C*4), Device::CUDA);
        cached_up_out = Tensor::view({(int)B, (int)T, (int)C*4}, scratchpad->get_address(B*T*C*4), Device::CUDA);
    }

    // SwiGLU: swish(gate(x)) * up(x)
    mlp_gate.forward_into(input, cached_gate_out);   // [B, T, 4C] raw gate
    mlp_up.forward_into(input, cached_up_out);       // [B, T, 4C] up projection
    
    int B_sw = cached_gate_out.shape[0], T_sw = cached_gate_out.shape[1], C4_sw = cached_gate_out.shape[2];
    size_t fwd_savepoint = 0;
    if (scratchpad && cached_gate_out.device == Device::CUDA) {
        fwd_savepoint = scratchpad->get_savepoint();
    }
    Tensor swiglu_tmp = (scratchpad && cached_gate_out.device == Device::CUDA)
        ? Tensor::view({B_sw, T_sw, C4_sw}, scratchpad->get_address((size_t)B_sw * T_sw * C4_sw), Device::CUDA)
        : cached_swiglu_tmp;
        
    Tensor::swiglu_forward(cached_gate_out, cached_up_out, swiglu_tmp); // FUSED kernel
    mlp_down.forward_into(swiglu_tmp, output);
    cached_out = output;
    
    if (scratchpad && cached_gate_out.device == Device::CUDA) {
        scratchpad->restore_savepoint(fwd_savepoint);
    }
}

Tensor SwiGLU::backward(const Tensor& dout) {
    backward_into(dout, cached_dX);
    return cached_dX;
}

void SwiGLU::backward_into(const Tensor& dout, Tensor& din) {
    int B = dout.shape[0];
    int T = dout.shape[1];
    int C = dout.shape[2];
    
    size_t bwd_savepoint = 0;
    if (scratchpad && dout.device == Device::CUDA) {
        bwd_savepoint = scratchpad->get_savepoint();
    }

    // Allocate scratch views for backward intermediates
    Tensor tmp_d_down = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({B, T, 4 * C}, scratchpad->get_address((size_t)B * T * 4 * C), Device::CUDA)
        : cached_d_down;
    Tensor tmp_d_gate = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({B, T, 4 * C}, scratchpad->get_address((size_t)B * T * 4 * C), Device::CUDA)
        : cached_d_gate;
    Tensor tmp_d_up = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({B, T, 4 * C}, scratchpad->get_address((size_t)B * T * 4 * C), Device::CUDA)
        : cached_d_up;
    Tensor tmp_d_gate_proj = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({B, T, C}, scratchpad->get_address((size_t)B * T * C), Device::CUDA)
        : cached_d_gate_proj;
    Tensor tmp_d_up_proj = (scratchpad && dout.device == Device::CUDA)
        ? Tensor::view({B, T, C}, scratchpad->get_address((size_t)B * T * C), Device::CUDA)
        : cached_d_up_proj;

    // Backprop through SwiGLU MLP branch
    mlp_down.backward_into(dout, tmp_d_down);                                      // d_down = [B,T,4C]
    Tensor::swiglu_backward(tmp_d_down, cached_up_out, cached_gate_out, tmp_d_up, tmp_d_gate); // FUSED kernel         // d_gate = swish'(gate_pre) * d_swish_in
    mlp_gate.backward_into(tmp_d_gate, tmp_d_gate_proj);                           // d_gate_proj = [B,T,C]
    mlp_up.backward_into(tmp_d_up, tmp_d_up_proj);                                 // d_up_proj = [B,T,C]
    
    Tensor::add_into(tmp_d_gate_proj, tmp_d_up_proj, din);                         // merge gate + up gradients
    cached_dX = din;
    
    if (scratchpad && dout.device == Device::CUDA) {
        scratchpad->restore_savepoint(bwd_savepoint);
    }
}

std::vector<Tensor*> SwiGLU::get_parameters() {
    std::vector<Tensor*> params;
    auto p_gate = mlp_gate.get_parameters(); params.insert(params.end(), p_gate.begin(), p_gate.end());
    auto p_up = mlp_up.get_parameters(); params.insert(params.end(), p_up.begin(), p_up.end());
    auto p_down = mlp_down.get_parameters(); params.insert(params.end(), p_down.begin(), p_down.end());
    return params;
}

#include <algorithm>
ScratchpadFootprint SwiGLU::get_footprint(int B, int T) {
    size_t C = mlp_gate.in_channels;
    
    // fwd_standing: cached_gate_out (4C) + cached_up_out (4C)
    size_t fwd_standing = (size_t)(8 * B * T * C);
    
    // fwd_temp: swiglu_tmp (4C) + children
    size_t fwd_temp = (size_t)(4 * B * T * C) + std::max<size_t>({
        mlp_gate.get_footprint(B,T).fwd_peak(), 
        mlp_up.get_footprint(B,T).fwd_peak(), 
        mlp_down.get_footprint(B,T).fwd_peak()
    });
    
    // bwd_temp: tmp_d_down (4C) + tmp_d_gate (4C) + tmp_d_up (4C) + tmp_d_gate_proj (1C) + tmp_d_up_proj (1C) = 14C
    size_t bwd_temp = (size_t)(14 * B * T * C) + std::max<size_t>({
        mlp_down.get_footprint(B,T).bwd_peak(), 
        mlp_gate.get_footprint(B,T).bwd_peak(), 
        mlp_up.get_footprint(B,T).bwd_peak()
    });
    
    return {fwd_standing, fwd_temp, bwd_temp};
}
