#include "Scratchpad.h"
#include "cuda_ops.h"
#include <string>
#include <iostream>

Scratchpad::Scratchpad(size_t max_floats, ScratchpadDevice dev)
    : buffer(nullptr), capacity_floats(max_floats), offset_floats(0), device(dev) {
    if (device == ScratchpadDevice::CUDA) {
        cuda_ops::allocate_memory(&buffer, capacity_floats);
    } else {
        buffer = new float[capacity_floats]();
    }
}

Scratchpad::~Scratchpad() {
    if (buffer) {
        if (device == ScratchpadDevice::CUDA) {
            cuda_ops::free_memory(buffer);
        } else {
            delete[] buffer;
        }
        buffer = nullptr;
    }
}

float* Scratchpad::get_address(size_t num_floats) {
    // 256-byte alignment => 64 floats alignment
    constexpr size_t ALIGN_FLOATS = 64;
    size_t aligned_num = (num_floats + ALIGN_FLOATS - 1) & ~(ALIGN_FLOATS - 1);
    if (offset_floats + aligned_num > capacity_floats) {
        size_t available_floats = (capacity_floats > offset_floats) ? (capacity_floats - offset_floats) : 0;
        std::cerr << "\n============================================================\n"
                  << " [FATAL ERROR] Scratchpad Memory Arena Out of Memory!\n"
                  << "============================================================\n"
                  << "  -> Requested Allocation: " << num_floats << " floats (~" 
                  << (num_floats * sizeof(float)) / (1024 * 1024) << " MB)\n"
                  << "  -> Available in Arena:   " << available_floats << " floats (~" 
                  << (available_floats * sizeof(float)) / (1024 * 1024) << " MB)\n"
                  << "  -> Total Arena Capacity: " << capacity_floats << " floats (~" 
                  << (capacity_floats * sizeof(float)) / (1024 * 1024) << " MB)\n"
                  << "  -> Current Offset:       " << offset_floats << " floats\n\n"
                  << "Please increase the scratchpad capacity passed to init_scratchpad(capacity_floats).\n"
                  << "============================================================\n";
        throw std::runtime_error("Scratchpad out of memory! Requested: " + std::to_string(num_floats) +
                                 " floats, Available: " + std::to_string(available_floats));
    }
    float* ptr = buffer + offset_floats;
    offset_floats += aligned_num;
    return ptr;
}

size_t Scratchpad::get_savepoint() const {
    return offset_floats;
}

void Scratchpad::restore_savepoint(size_t savepoint) {
    if (savepoint > offset_floats) {
        throw std::runtime_error("Invalid savepoint restore: savepoint exceeds current offset.");
    }
    offset_floats = savepoint;
}

void Scratchpad::reset() {
    offset_floats = 0;
}
