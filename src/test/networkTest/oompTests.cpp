#include <iostream>
#include <cassert>
#include "GPT.h"
#include "LinearLayer.h"
#include "TransformerBlock.h"

void test_linear_footprint() {
    LinearLayer linear(128, 256);
    auto fp = linear.get_footprint(32, 64);
    size_t expected_bwd = 256 + 128 * 32 * 64 + 2 * 128 * 256;
    assert(fp.fwd_standing == 0);
    assert(fp.fwd_peak == 0);
    assert(fp.bwd_peak == expected_bwd);
    std::cout << "LinearLayer OOMP test passed!\n";
}

void test_gpt_footprint() {
    GPTConfig config;
    config.vocab_size = 22;
    config.max_seq_len = 512;
    config.embed_dim = 320;
    config.num_layers = 8;
    config.use_gradient_checkpointing = true;
    config.num_heads = 10; // Default usually embed_dim / 32 = 10
    config.use_flash_attention = false;
    
    GPT gpt(config);
    auto fp = gpt.get_footprint(16, 512);
    auto fp_mha = gpt.blocks[0]->attn.get_footprint(16, 512);
    auto fp_moe = gpt.blocks[0]->mlp.get_footprint(16, 512);
    auto fp_block = gpt.blocks[0]->get_footprint(16, 512);
    
    std::cout << "MHA Footprint: fwd_standing=" << fp_mha.fwd_standing << ", fwd_peak=" << fp_mha.fwd_peak() << ", bwd_peak=" << fp_mha.bwd_peak() << "\n";
    std::cout << "MoE Footprint: fwd_standing=" << fp_moe.fwd_standing << ", fwd_peak=" << fp_moe.fwd_peak() << ", bwd_peak=" << fp_moe.bwd_peak() << "\n";
    std::cout << "Block Footprint: fwd_standing=" << fp_block.fwd_standing << ", fwd_peak=" << fp_block.fwd_peak() << ", bwd_peak=" << fp_block.bwd_peak() << "\n";
    std::cout << "GPT Footprint: fwd_standing=" << fp.fwd_standing << ", fwd_peak=" << fp.fwd_peak() << ", bwd_peak=" << fp.bwd_peak() << "\n";
    
    // Allocate the scratchpad with exactly what OOMP predicts
    size_t peak_floats = std::max(fp.fwd_peak(), fp.bwd_peak());
    gpt.init_scratchpad(peak_floats);
    
    std::cout << "Predicted Peak Floats: " << peak_floats << "\n";
    
    Tensor input_ids({16, 512}, 0.0f, Device::CUDA);
    Tensor output;
    
    for (int i = 0; i < 5; i++) {
        std::cout << "Iteration " << i << " Running Forward..." << std::endl;
        gpt.to(Device::CUDA);
        gpt.forward_into(input_ids, output);
        
        Tensor dout({16, 512, 22}, 0.1f, Device::CUDA);
        Tensor din;
        
        std::cout << "Iteration " << i << " Running Backward..." << std::endl;
        gpt.backward_into(dout, din);
    }
    
    std::cout << "GPT OOMP test passed!\n";
}

int main() {
    test_linear_footprint();
    test_gpt_footprint();
    std::cout << "All OOMP tests passed!\n";
    return 0;
}
