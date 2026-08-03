#include "MoELayer.h"
#include "Tensor.h"
#include <iostream>
#include <cassert>

void test_forward() {
    int B = 2, T = 4, C = 16;
    int num_experts = 4, top_k = 2;
    
    MoELayer moe(C, -1, num_experts, top_k);
    Tensor input({B, T, C}, 1.0f, Device::CPU);
    
    Tensor out = moe.forward(input);
    
    assert(out.shape.size() == 3);
    assert(out.shape[0] == B);
    assert(out.shape[1] == T);
    assert(out.shape[2] == C);
    std::cout << "Forward pass OK" << std::endl;
}

void test_backward() {
    int B = 2, T = 4, C = 16;
    int num_experts = 4, top_k = 2;
    
    MoELayer moe(C, -1, num_experts, top_k);
    Tensor input({B, T, C}, 1.0f, Device::CPU);
    
    moe.forward(input);
    
    Tensor dout({B, T, C}, 1.0f, Device::CPU);
    Tensor din = moe.backward(dout);
    
    assert(din.shape.size() == 3);
    assert(din.shape[0] == B);
    assert(din.shape[1] == T);
    assert(din.shape[2] == C);
    std::cout << "Backward pass OK" << std::endl;
}

int main() {
    test_forward();
    test_backward();
    
    // Quick test if CUDA is available
#ifdef USE_CUDA
    std::cout << "Starting test_forward_cuda..." << std::endl;
#endif
        int B = 2, T = 4, C = 16;
        int num_experts = 4, top_k = 2;
        
        MoELayer moe(C, -1, num_experts, top_k);
        moe.to(Device::CUDA);
        Tensor input({B, T, C}, 1.0f, Device::CUDA);
        
        Tensor out = moe.forward(input);
        
        assert(out.shape.size() == 3);
        assert(out.shape[0] == B);
        assert(out.shape[1] == T);
        assert(out.shape[2] == C);
        std::cout << "Forward CUDA pass OK" << std::endl;
        
        Tensor dout({B, T, C}, 1.0f, Device::CUDA);
        Tensor din = moe.backward(dout);
        
        assert(din.shape.size() == 3);
        assert(din.shape[0] == B);
        assert(din.shape[1] == T);
        assert(din.shape[2] == C);
        std::cout << "Backward CUDA pass OK" << std::endl;
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
