#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <vector>

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(err) << "\n";
        exit(1);
    }
}

int main() {
    int iterations = 10;
    int allocations_per_step = 400; // Average allocations in 16-layer model per step
    std::vector<size_t> sizes(allocations_per_step);
    
    // Simulate typical allocation sizes ranging from 1MB to 10MB
    for(int i = 0; i < allocations_per_step; ++i) {
        sizes[i] = ((rand() % 10) + 1) * 1024 * 1024;
    }

    std::cout << "Benchmarking dynamic cudaMalloc / cudaFree thrashing...\n";
    std::cout << "Simulating " << allocations_per_step << " allocations per step for " << iterations << " steps.\n";

    // Warmup
    float* tmp;
    cudaMalloc((void**)&tmp, 1024);
    cudaFree(tmp);
    cudaDeviceSynchronize();

    std::vector<float*> ptrs(allocations_per_step, nullptr);

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < iterations; step++) {
        // Allocate
        for (int i = 0; i < allocations_per_step; i++) {
            cudaMalloc((void**)&ptrs[i], sizes[i]);
        }
        // Free
        for (int i = 0; i < allocations_per_step; i++) {
            cudaFree(ptrs[i]);
        }
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    
    double ms_per_step = diff.count() / iterations;
    
    std::cout << "\n--- RESULTS ---\n";
    std::cout << "Average overhead per training step: " << ms_per_step << " ms\n";
    std::cout << "Total overhead eliminated over a 72,000-step training run: " << (ms_per_step * 72000.0) / (1000.0 * 60.0 * 60.0) << " HOURS\n";

    return 0;
}
