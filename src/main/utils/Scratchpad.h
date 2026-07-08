#ifndef SCRATCHPAD_H
#define SCRATCHPAD_H

#include <cstddef>
#include <stdexcept>
#include <iostream>

enum class ScratchpadDevice {
    CPU,
    CUDA
};

class Scratchpad {
private:
    float* buffer;
    size_t capacity_floats;
    size_t offset_floats;
    ScratchpadDevice device;

public:
    explicit Scratchpad(size_t max_floats, ScratchpadDevice dev = ScratchpadDevice::CPU);
    ~Scratchpad();

    // Prevent copying
    Scratchpad(const Scratchpad&) = delete;
    Scratchpad& operator=(const Scratchpad&) = delete;

    // Allocate num_floats aligned to 256 bytes (64 floats)
    float* get_address(size_t num_floats);

    // Savepoint management for stack-like push/pop allocation
    size_t get_savepoint() const;
    void restore_savepoint(size_t savepoint);

    // Reset arena offset to 0
    void reset();

    size_t get_capacity() const { return capacity_floats; }
    size_t get_offset() const { return offset_floats; }
    ScratchpadDevice get_device() const { return device; }
};

#endif // SCRATCHPAD_H
