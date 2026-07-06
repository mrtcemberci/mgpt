#include "Tensor.h"
#include <algorithm> // for std::fill
#include <iostream>
#include <bits/ostream.tcc>

Tensor::Tensor(const std::vector<int>& dims, float init_val) : shape(dims) {
    size_t total_size = 1;
    for (int dim : shape) {
        total_size *= dim;
    }

    data = std::vector<float>(total_size, init_val);

    grad = std::vector<float>(total_size, 0.0f);
}

size_t Tensor::size() const {
    return data.size();
}

void Tensor::zero_grad() {
    std::fill(grad.begin(), grad.end(), 0.0f);
}

size_t Tensor::offset(const std::vector<int>& indices) const {
    size_t flat_idx = 0;
    size_t stride = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        flat_idx += indices[i] * stride;
        stride *= shape[i];
    }
    return flat_idx;
}

Tensor Tensor::applyOperation(const Tensor &other, const std::function<float(float, float)>& operation) const {
    if (other.shape != shape) {
        std::cerr << "Tensor::applyOperation: shape mismatch" << std::endl;
        exit(-1);
    }

    Tensor result = Tensor(shape, 0.0f);

    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = operation(data[i], other.data[i]);
    }

    return result;
}

Tensor Tensor::operator+(const Tensor& other) const {
    return applyOperation(other, [](float a, float b) { return a + b; });
}

Tensor Tensor::operator*(const Tensor& other) const {
    return applyOperation(other, [](float a, float b) { return a * b; });
}

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    size_t new_size = 1;
    for (int dim : new_shape) {
        new_size *= dim;
    }
    if (new_size != data.size()) {
        std::cerr << "Tensor::reshape: Total elements must remain identical!" << std::endl;
        exit(-1);
    }
    Tensor result(new_shape, 0.0f);
    result.data = data;
    result.grad = grad;
    return result;
}

Tensor Tensor::transpose(int dim1, int dim2) const {
    if (dim1 < 0 || dim1 >= shape.size() || dim2 < 0 || dim2 >= shape.size()) {
        std::cerr << "Tensor::transpose: Invalid dimension indices" << std::endl;
        exit(-1);
    }
    std::vector<int> new_shape = shape;
    std::swap(new_shape[dim1], new_shape[dim2]);

    Tensor result(new_shape, 0.0f);

    for (size_t i = 0; i < data.size(); ++i) {
        size_t temp = i;
        std::vector<int> coord(shape.size());
        for (int d = (int)shape.size() - 1; d >= 0; --d) {
            coord[d] = temp % shape[d];
            temp /= shape[d];
        }

        std::swap(coord[dim1], coord[dim2]);

        size_t dest_idx = result.offset(coord);
        result.data[dest_idx] = data[i];
    }
    return result;
}

void Tensor::matmul2d_raw(const float* A, const float* B, float* C, int M, int K, int N) const {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a_val = A[i * K + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
    }
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (shape.size() < 2 || other.shape.size() < 2) {
        std::cerr << "matmul: both tensors must have at least 2 dimensions" << std::endl;
        exit(-1);
    }

    int K = shape.back();
    int M = shape[shape.size() - 2];
    int K2 = other.shape[other.shape.size() - 2];
    int N = other.shape.back();

    if (K != K2) {
        std::cerr << "matmul: inner dimensions must match (" << K << " vs " << K2 << ")" << std::endl;
        exit(-1);
    }

    if (other.shape.size() == 2) {
        std::vector<int> out_shape = shape;
        out_shape.back() = N; // Replace last dimension K with N
        Tensor result(out_shape, 0.0f);

        int total_rows = (int)(data.size() / K);
        matmul2d_raw(data.data(), other.data.data(), result.data.data(), total_rows, K, N);
        return result;
    }

    if (shape.size() == other.shape.size()) {
        std::vector<int> out_shape = shape;
        out_shape.back() = N;
        int num_batches = 1;
        for (size_t i = 0; i < shape.size() - 2; ++i) {
            if (shape[i] != other.shape[i]) {
                std::cerr << "matmul: outer batch dimensions must match" << std::endl;
                exit(-1);
            }
            num_batches *= shape[i];
        }

        Tensor result(out_shape, 0.0f);
        for (int b = 0; b < num_batches; ++b) {
            const float* a_ptr = data.data() + b * (M * K);
            const float* b_ptr = other.data.data() + b * (K * N);
            float* c_ptr = result.data.data() + b * (M * N);
            matmul2d_raw(a_ptr, b_ptr, c_ptr, M, K, N);
        }
        return result;
    }

    std::cerr << "matmul: unsupported shape combination" << std::endl;
    exit(-1);
}

