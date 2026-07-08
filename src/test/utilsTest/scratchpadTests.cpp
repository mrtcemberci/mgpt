#include "Scratchpad.h"
#include "Tensor.h"
#include <iostream>
#include <cassert>
#include <stdexcept>

void test_scratchpad_cpu() {
    std::cout << "Running test_scratchpad_cpu..." << std::endl;
    Scratchpad pad(1000, ScratchpadDevice::CPU);
    assert(pad.get_capacity() == 1000);
    assert(pad.get_offset() == 0);

    float* ptr1 = pad.get_address(100);
    assert(ptr1 != nullptr);
    assert(pad.get_offset() == 128);

    size_t savepoint = pad.get_savepoint();

    float* ptr2 = pad.get_address(50);
    assert(ptr2 != nullptr);
    assert(pad.get_offset() == 192);

    pad.restore_savepoint(savepoint);
    assert(pad.get_offset() == 128);

    pad.reset();
    assert(pad.get_offset() == 0);
    std::cout << "  -> test_scratchpad_cpu passed! ✅" << std::endl;
}

void test_tensor_view_cuda() {
    std::cout << "Running test_tensor_view_cuda..." << std::endl;
    // 1. Verify precondition: dev must be Device::CUDA
    bool threw_error = false;
    try {
        Tensor::view({10, 10}, (float*)0x1234, Device::CPU);
    } catch (const std::runtime_error&) {
        threw_error = true;
    }
    assert(threw_error && "Tensor::view should throw runtime_error when Device::CPU is passed");

    // 2. Test CUDA Scratchpad + Tensor::view lifecycle
    Scratchpad pad(1024, ScratchpadDevice::CUDA);
    float* vram_ptr = pad.get_address(100);
    assert(vram_ptr != nullptr);

    {
        Tensor view_tensor = Tensor::view({10, 10}, vram_ptr, Device::CUDA);
        assert(view_tensor.is_owning == false);
        assert(view_tensor.get_data_ptr() == vram_ptr);
        assert(view_tensor.size() == 100);
        // Destructor ~Tensor() will run here and MUST NOT free vram_ptr
    }

    // Verify vram_ptr arena address remains intact and pad can continue allocating
    float* next_ptr = pad.get_address(50);
    assert(next_ptr != nullptr);
    std::cout << "  -> test_tensor_view_cuda passed! ✅" << std::endl;
}

int main() {
    test_scratchpad_cpu();
    test_tensor_view_cuda();
    return 0;
}
