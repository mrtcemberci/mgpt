#include "Tensor.h"
#include <algorithm> // for std::fill
#include <iostream>
#include <random>
#include <cmath>
#include <bits/ostream.tcc>

Tensor::Tensor(const std::vector<int>& dims, float init_val) : shape(dims) {
    size_t total_size = 1;
    for (int dim : shape) {
        total_size *= dim;
    }

    data = std::vector<float>(total_size, init_val);

    grad = std::vector<float>(total_size, 0.0f);
}

Tensor Tensor::randn(const std::vector<int>& dims, float mean, float stddev) {
    Tensor t(dims, 0.0f);
    t.randomize(mean, stddev);
    return t;
}

void Tensor::randomize(float mean, float stddev) {
    static std::mt19937 gen(1337);
    std::normal_distribution<float> dist(mean, stddev);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(gen);
    }
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
        if (!shape.empty() && other.size() == shape.back()) {
            Tensor result = Tensor(shape, 0.0f);
            int C = shape.back();
            int total_rows = (int)(data.size() / C);
            for (int i = 0; i < total_rows; ++i) {
                for (int j = 0; j < C; ++j) {
                    result.data[i * C + j] = operation(data[i * C + j], other.data[j]);
                }
            }
            return result;
        }
        // Check if `other` is a single-element scalar tensor
        if (other.size() == 1) {
            float val = other.data[0];
            return map([&](float a) { return operation(a, val); });
        }
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
    if (other.shape != shape) {
        if (!shape.empty() && other.size() == shape.back()) {
            Tensor result(shape, 0.0f);
            int C = shape.back();
            int total_rows = (int)(data.size() / C);
            for (int i = 0; i < total_rows; ++i) {
                for (int j = 0; j < C; ++j) {
                    result.data[i * C + j] = data[i * C + j] + other.data[j];
                }
            }
            return result;
        }
        if (other.size() == 1) {
            return (*this) + other.data[0];
        }
        std::cerr << "Tensor::operator+: shape mismatch" << std::endl;
        exit(-1);
    }
    Tensor result(shape, 0.0f);
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (other.shape != shape) {
        if (!shape.empty() && other.size() == shape.back()) {
            Tensor result(shape, 0.0f);
            int C = shape.back();
            int total_rows = (int)(data.size() / C);
            for (int i = 0; i < total_rows; ++i) {
                for (int j = 0; j < C; ++j) {
                    result.data[i * C + j] = data[i * C + j] - other.data[j];
                }
            }
            return result;
        }
        if (other.size() == 1) {
            return (*this) - other.data[0];
        }
        std::cerr << "Tensor::operator-: shape mismatch" << std::endl;
        exit(-1);
    }
    Tensor result(shape, 0.0f);
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (other.shape != shape) {
        if (!shape.empty() && other.size() == shape.back()) {
            Tensor result(shape, 0.0f);
            int C = shape.back();
            int total_rows = (int)(data.size() / C);
            for (int i = 0; i < total_rows; ++i) {
                for (int j = 0; j < C; ++j) {
                    result.data[i * C + j] = data[i * C + j] * other.data[j];
                }
            }
            return result;
        }
        if (other.size() == 1) {
            return (*this) * other.data[0];
        }
        std::cerr << "Tensor::operator*: shape mismatch" << std::endl;
        exit(-1);
    }
    Tensor result(shape, 0.0f);
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

Tensor Tensor::map(const std::function<float(float)>& func) const {
    Tensor result(shape, 0.0f);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = func(data[i]);
    }
    return result;
}

Tensor Tensor::operator+(float scalar) const {
    Tensor result(shape, 0.0f);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + scalar;
    }
    return result;
}

Tensor Tensor::operator-(float scalar) const {
    Tensor result(shape, 0.0f);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] - scalar;
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape, 0.0f);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * scalar;
    }
    return result;
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
    if (dim1 < 0 || dim1 >= (int)shape.size() || dim2 < 0 || dim2 >= (int)shape.size()) {
        std::cerr << "Tensor::transpose: Invalid dimension indices" << std::endl;
        exit(-1);
    }
    std::vector<int> new_shape = shape;
    std::swap(new_shape[dim1], new_shape[dim2]);

    Tensor result(new_shape, 0.0f);

    // Fast path for 2D matrix transpose
    if (shape.size() == 2 && ((dim1 == 0 && dim2 == 1) || (dim1 == 1 && dim2 == 0))) {
        int M = shape[0];
        int N = shape[1];
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                result.data[j * M + i] = data[i * N + j];
            }
        }
        return result;
    }

    // Fast path for 3D tensor transpose (e.g., swapping time T and channel C: {B, T, C} <-> {B, C, T})
    if (shape.size() == 3 && ((dim1 == 1 && dim2 == 2) || (dim1 == 2 && dim2 == 1))) {
        int B = shape[0];
        int T = shape[1];
        int C = shape[2];
        for (int b = 0; b < B; ++b) {
            const float* src = data.data() + b * (T * C);
            float* dst = result.data.data() + b * (C * T);
            for (int t = 0; t < T; ++t) {
                for (int c = 0; c < C; ++c) {
                    dst[c * T + t] = src[t * C + c];
                }
            }
        }
        return result;
    }

    // Fallback for N-D tensors without allocating memory inside the loop
    std::vector<int> coord(shape.size());
    for (size_t i = 0; i < data.size(); ++i) {
        size_t temp = i;
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
        out_shape.back() = N;
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

