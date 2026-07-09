#include "Tensor.h"
#include "cuda_ops.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <cmath>
#include <stdexcept>
#include <cassert>

Tensor::~Tensor() {
    if (is_owning && device == Device::CUDA) {
        cuda_ops::free_memory(cuda_data);
        cuda_ops::free_memory(cuda_grad);
    }
}

Tensor::Tensor(const Tensor& other)
    : device(other.device), is_owning(other.is_owning), data(other.data), grad(other.grad), shape(other.shape), cuda_data(nullptr), cuda_grad(nullptr) {
    if (device == Device::CUDA) {
        if (!is_owning) {
            cuda_data = other.cuda_data;
            cuda_grad = other.cuda_grad;
        } else {
            size_t sz = size();
            if (other.cuda_data && sz > 0) {
                cuda_ops::allocate_memory(&cuda_data, sz);
                cuda_ops::copy_device_to_device(cuda_data, other.cuda_data, sz);
            }
            if (other.cuda_grad && sz > 0) {
                cuda_ops::allocate_memory(&cuda_grad, sz);
                cuda_ops::copy_device_to_device(cuda_grad, other.cuda_grad, sz);
            }
        }
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : device(other.device), is_owning(other.is_owning), data(std::move(other.data)), grad(std::move(other.grad)),
      cuda_data(other.cuda_data), cuda_grad(other.cuda_grad), shape(std::move(other.shape)) {
    if (other.is_owning) {
        other.cuda_data = nullptr;
        other.cuda_grad = nullptr;
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;
    size_t new_sz = 1;
    for (int dim : other.shape) new_sz *= dim;
    if (other.shape.empty()) new_sz = 0;

    if (is_owning && device == Device::CUDA && other.is_owning && other.device == Device::CUDA && size() == new_sz && new_sz > 0 && cuda_data) {
        shape = other.shape;
        if (other.cuda_data) {
            cuda_ops::copy_device_to_device(cuda_data, other.cuda_data, new_sz);
        }
        if (other.cuda_grad) {
            if (!cuda_grad) cuda_ops::allocate_memory(&cuda_grad, new_sz);
            cuda_ops::copy_device_to_device(cuda_grad, other.cuda_grad, new_sz);
        }
        return *this;
    }

    if (is_owning && device == Device::CUDA) {
        cuda_ops::free_memory(cuda_data);
        cuda_ops::free_memory(cuda_grad);
    }
    device = other.device;
    is_owning = other.is_owning;
    data = other.data;
    grad = other.grad;
    shape = other.shape;
    cuda_data = nullptr;
    cuda_grad = nullptr;
    if (device == Device::CUDA) {
        if (!is_owning) {
            cuda_data = other.cuda_data;
            cuda_grad = other.cuda_grad;
        } else {
            size_t sz = size();
            if (other.cuda_data && sz > 0) {
                cuda_ops::allocate_memory(&cuda_data, sz);
                cuda_ops::copy_device_to_device(cuda_data, other.cuda_data, sz);
            }
            if (other.cuda_grad && sz > 0) {
                cuda_ops::allocate_memory(&cuda_grad, sz);
                cuda_ops::copy_device_to_device(cuda_grad, other.cuda_grad, sz);
            }
        }
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;
    if (is_owning && device == Device::CUDA) {
        cuda_ops::free_memory(cuda_data);
        cuda_ops::free_memory(cuda_grad);
    }
    device = other.device;
    is_owning = other.is_owning;
    data = std::move(other.data);
    grad = std::move(other.grad);
    cuda_data = other.cuda_data;
    cuda_grad = other.cuda_grad;
    shape = std::move(other.shape);
    if (other.is_owning) {
        other.cuda_data = nullptr;
        other.cuda_grad = nullptr;
    }
    return *this;
}

Tensor Tensor::view(const std::vector<int>& dims, float* vram_ptr, Device dev) {
    if (dev != Device::CUDA) {
        throw std::runtime_error("Tensor::view precondition failed: dev must be Device::CUDA (VRAM address only)!");
    }
    if (!vram_ptr) {
        throw std::runtime_error("Tensor::view precondition failed: vram_ptr cannot be null!");
    }
    Tensor t;
    t.device = dev;
    t.shape = dims;
    t.cuda_data = vram_ptr;
    t.is_owning = false;
    return t;
}

Tensor::Tensor(const std::vector<int>& dims, float init_val, Device dev) : shape(dims), device(dev) {
    size_t total_size = 1;
    for (int dim : shape) {
        if (dim <= 0) throw std::invalid_argument("Tensor dimensions must be positive.");
        total_size *= dim;
    }
    if (device == Device::CUDA) {
        cuda_ops::allocate_memory(&cuda_data, total_size);
        cuda_ops::fill(cuda_data, init_val, total_size);
    } else {
        data = std::vector<float>(total_size, init_val);
        grad = std::vector<float>(total_size, 0.0f);
    }
}

Tensor Tensor::randn(const std::vector<int>& dims, float mean, float stddev, Device dev) {
    Tensor t(dims, 0.0f, dev);
    t.randomize(mean, stddev);
    return t;
}

void Tensor::randomize(float mean, float stddev) {
    static std::mt19937 gen(1337);
    std::normal_distribution<float> dist(mean, stddev);
    size_t sz = size();
    if (device == Device::CUDA) {
        std::vector<float> tmp(sz);
        for (size_t i = 0; i < sz; ++i) tmp[i] = dist(gen);
        cuda_ops::copy_host_to_device(cuda_data, tmp.data(), sz);
    } else {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = dist(gen);
        }
    }
}

size_t Tensor::size() const {
    if (shape.empty()) return 0;
    size_t total_size = 1;
    for (int dim : shape) total_size *= dim;
    return total_size;
}

void Tensor::zero_grad() {
    if (device == Device::CUDA) {
        if (cuda_grad) cuda_ops::fill(cuda_grad, 0.0f, size());
    } else {
        std::fill(grad.begin(), grad.end(), 0.0f);
    }
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

Tensor& Tensor::to(Device target_device) {
    if (device == target_device) return *this;
    size_t sz = size();
    if (target_device == Device::CUDA) {
        cuda_ops::allocate_memory(&cuda_data, sz);
        if (!data.empty()) {
            cuda_ops::copy_host_to_device(cuda_data, data.data(), sz);
            data.clear();
        }
        if (!grad.empty()) {
            cuda_ops::allocate_memory(&cuda_grad, sz);
            cuda_ops::copy_host_to_device(cuda_grad, grad.data(), sz);
            grad.clear();
        }
    } else { // target_device == Device::CPU
        data.resize(sz, 0.0f);
        if (cuda_data) {
            cuda_ops::copy_device_to_host(data.data(), cuda_data, sz);
            cuda_ops::free_memory(cuda_data);
            cuda_data = nullptr;
        }
        if (cuda_grad) {
            grad.resize(sz, 0.0f);
            cuda_ops::copy_device_to_host(grad.data(), cuda_grad, sz);
            cuda_ops::free_memory(cuda_grad);
            cuda_grad = nullptr;
        }
    }
    device = target_device;
    return *this;
}

float* Tensor::get_data_ptr() {
    return device == Device::CUDA ? cuda_data : data.data();
}

const float* Tensor::get_data_ptr() const {
    return device == Device::CUDA ? cuda_data : data.data();
}

float* Tensor::get_grad_ptr() {
    if (device == Device::CUDA) {
        if (!cuda_grad && size() > 0) {
            cuda_ops::allocate_memory(&cuda_grad, size());
            cuda_ops::fill(cuda_grad, 0.0f, size());
        }
        return cuda_grad;
    }
    if (grad.empty() && size() > 0) grad.resize(size(), 0.0f);
    return grad.data();
}

const float* Tensor::get_grad_ptr() const {
    return device == Device::CUDA ? cuda_grad : grad.data();
}

void Tensor::copy_from_host(const float* src) {
    size_t sz = size();
    if (sz == 0) return;
    if (device == Device::CUDA) {
        if (!cuda_data) cuda_ops::allocate_memory(&cuda_data, sz);
        cuda_ops::copy_host_to_device(cuda_data, src, sz);
    } else {
        if (data.size() < sz) data.resize(sz);
        std::copy(src, src + sz, data.begin());
    }
}

void Tensor::copy_to_host(float* dst) const {
    size_t sz = size();
    if (sz == 0) return;
    if (device == Device::CUDA) {
        if (cuda_data) cuda_ops::copy_device_to_host(dst, cuda_data, sz);
    } else {
        std::copy(data.begin(), data.end(), dst);
    }
}

Tensor Tensor::applyOperation(const Tensor &other, const std::function<float(float, float)>& operation) const {
    if (other.shape != shape) {
        if (!shape.empty() && other.size() == shape.back()) {
            Tensor result = Tensor(shape, 0.0f, device);
            int C = shape.back();
            int total_rows = (int)(size() / C);
            for (int i = 0; i < total_rows; ++i) {
                for (int j = 0; j < C; ++j) {
                    result.data[i * C + j] = operation(data[i * C + j], other.data[j]);
                }
            }
            return result;
        }
        if (other.size() == 1) {
            float val = 0.0f;
            if (other.device == Device::CUDA) cuda_ops::copy_device_to_host(&val, other.cuda_data, 1);
            else val = other.data[0];
            return map([&](float a) { return operation(a, val); });
        }
        std::cerr << "Tensor::applyOperation: shape mismatch" << std::endl;
        exit(-1);
    }

    Tensor result = Tensor(shape, 0.0f, device);
    for (size_t i = 0; i < size(); i++) {
        result.data[i] = operation(data[i], other.data[i]);
    }
    return result;
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (device == Device::CUDA) {
        if (other.shape == shape) {
            Tensor result(shape, 0.0f, Device::CUDA);
            cuda_ops::add(get_data_ptr(), other.get_data_ptr(), result.get_data_ptr(), size());
            return result;
        }
        if (!shape.empty() && other.size() == shape.back()) {
            Tensor result(shape, 0.0f, Device::CUDA);
            int C = shape.back();
            int total_rows = (int)(size() / C);
            cuda_ops::add_broadcast(get_data_ptr(), other.get_data_ptr(), result.get_data_ptr(), total_rows, C);
            return result;
        }
        if (other.size() == 1) {
            float val = 0.0f;
            if (other.device == Device::CUDA) cuda_ops::copy_device_to_host(&val, other.cuda_data, 1);
            else val = other.data[0];
            return (*this) + val;
        }
        std::cerr << "Tensor::operator+: CUDA shape mismatch" << std::endl;
        exit(-1);
    }

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

    return applyOperation(other, [](float a, float b) { return a + b; });
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (device == Device::CUDA) {
        if (other.shape == shape) {
            Tensor result(shape, 0.0f, Device::CUDA);
            cuda_ops::sub(get_data_ptr(), other.get_data_ptr(), result.get_data_ptr(), size());
            return result;
        }
        if (other.size() == 1) {
            float val = 0.0f;
            if (other.device == Device::CUDA) cuda_ops::copy_device_to_host(&val, other.cuda_data, 1);
            else val = other.data[0];
            return (*this) - val;
        }
        std::cerr << "Tensor::operator-: CUDA shape mismatch" << std::endl;
        exit(-1);
    }

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

    return applyOperation(other, [](float a, float b) { return a - b; });
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (device == Device::CUDA) {
        if (other.shape == shape) {
            Tensor result(shape, 0.0f, Device::CUDA);
            cuda_ops::mul(get_data_ptr(), other.get_data_ptr(), result.get_data_ptr(), size());
            return result;
        }
        if (other.size() == 1) {
            float val = 0.0f;
            if (other.device == Device::CUDA) cuda_ops::copy_device_to_host(&val, other.cuda_data, 1);
            else val = other.data[0];
            return (*this) * val;
        }
        std::cerr << "Tensor::operator*: CUDA shape mismatch" << std::endl;
        exit(-1);
    }

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

    return applyOperation(other, [](float a, float b) { return a * b; });
}

Tensor Tensor::map(const std::function<float(float)>& func) const {
    Tensor result = Tensor(shape, 0.0f, device);
    if (device == Device::CUDA) {
        std::vector<float> tmp(size());
        cuda_ops::copy_device_to_host(tmp.data(), cuda_data, size());
        for (size_t i = 0; i < tmp.size(); i++) {
            tmp[i] = func(tmp[i]);
        }
        cuda_ops::copy_host_to_device(result.cuda_data, tmp.data(), size());
        return result;
    }
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = func(data[i]);
    }
    return result;
}

Tensor Tensor::operator+(float scalar) const {
    Tensor result = Tensor(shape, 0.0f, device);
    if (device == Device::CUDA) {
        cuda_ops::add_scalar(get_data_ptr(), scalar, result.get_data_ptr(), size());
        return result;
    }
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] + scalar;
    }
    return result;
}

Tensor Tensor::operator-(float scalar) const {
    Tensor result = Tensor(shape, 0.0f, device);
    if (device == Device::CUDA) {
        cuda_ops::sub_scalar(get_data_ptr(), scalar, result.get_data_ptr(), size());
        return result;
    }
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] - scalar;
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result = Tensor(shape, 0.0f, device);
    if (device == Device::CUDA) {
        cuda_ops::mul_scalar(get_data_ptr(), scalar, result.get_data_ptr(), size());
        return result;
    }
    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    size_t new_size = 1;
    for (int dim : new_shape) {
        new_size *= dim;
    }
    if (new_size != size()) {
        std::cerr << "Tensor::reshape: Total elements must remain identical!" << std::endl;
        exit(-1);
    }
    Tensor result(new_shape, 0.0f, device);
    if (device == Device::CUDA) {
        if (cuda_data) cuda_ops::copy_device_to_device(result.cuda_data, cuda_data, size());
        if (cuda_grad && result.cuda_grad) cuda_ops::copy_device_to_device(result.cuda_grad, cuda_grad, size());
    } else {
        result.data = data;
        result.grad = grad;
    }
    return result;
}

Tensor Tensor::transpose(int dim1, int dim2) const {
    if (dim1 < 0 || dim1 >= (int)shape.size() || dim2 < 0 || dim2 >= (int)shape.size()) {
        std::cerr << "Tensor::transpose: Invalid dimension indices" << std::endl;
        exit(-1);
    }
    std::vector<int> new_shape = shape;
    std::swap(new_shape[dim1], new_shape[dim2]);

    Tensor result(new_shape, 0.0f, device);
    if (device == Device::CUDA) {
        if (shape.size() == 2) {
            cuda_ops::transpose_2d(get_data_ptr(), result.get_data_ptr(), shape[0], shape[1]);
        } else if (shape.size() == 3) {
            cuda_ops::transpose_3d(get_data_ptr(), result.get_data_ptr(), shape[0], shape[1], shape[2], dim1, dim2);
        } else {
            std::cerr << "CUDA transpose only supports 2D and 3D tensors currently!" << std::endl;
            exit(-1);
        }
        return result;
    }

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

    std::vector<int> coord(shape.size());
    for (size_t i = 0; i < data.size(); ++i) {
        size_t temp = i;
        for (int d = (int)shape.size() - 1; d >= 0; --d) {
            coord[d] = (int)(temp % shape[d]);
            temp /= shape[d];
        }
        std::vector<int> new_coord = coord;
        std::swap(new_coord[dim1], new_coord[dim2]);
        size_t new_idx = result.offset(new_coord);
        result.data[new_idx] = data[i];
    }
    return result;
}

void Tensor::matmul2d_raw(const float* A, const float* B, float* C, int M, int K, int N) const {
    std::fill(C, C + (size_t)M * N, 0.0f);
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
        std::cerr << "matmul: inner dimensions must match: " << K << " vs " << K2 << std::endl;
        exit(-1);
    }

    if (other.shape.size() == 2) {
        std::vector<int> out_shape = shape;
        out_shape.back() = N;
        int total_rows = (int)(size() / K);
        Tensor result(out_shape, 0.0f, device);
        if (device == Device::CUDA) {
            cuda_ops::matmul(get_data_ptr(), other.get_data_ptr(), result.get_data_ptr(), total_rows, K, N, 1);
        } else {
            matmul2d_raw(data.data(), other.data.data(), result.data.data(), total_rows, K, N);
        }
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

        Tensor result(out_shape, 0.0f, device);
        if (device == Device::CUDA) {
            cuda_ops::matmul(get_data_ptr(), other.get_data_ptr(), result.get_data_ptr(), M, K, N, num_batches);
        } else {
            for (int b = 0; b < num_batches; ++b) {
                const float* a_ptr = data.data() + b * (M * K);
                const float* b_ptr = other.data.data() + b * (K * N);
                float* c_ptr = result.data.data() + b * (M * N);
                matmul2d_raw(a_ptr, b_ptr, c_ptr, M, K, N);
            }
        }
        return result;
    }

    std::cerr << "matmul: unsupported shape combination" << std::endl;
    exit(-1);
}

void Tensor::add_grad(const Tensor& dgrad) {
    if (size() != dgrad.size()) {
        std::cerr << "add_grad: size mismatch" << std::endl;
        exit(-1);
    }
    if (device == Device::CUDA) {
        cuda_ops::add(get_grad_ptr(), dgrad.get_data_ptr(), get_grad_ptr(), size());
        return;
    }
    for (size_t i = 0; i < this->grad.size(); ++i) {
        this->grad[i] += dgrad.data[i];
    }
}

Tensor Tensor::sum_rows() const {
    if (shape.size() < 2) {
        std::cerr << "sum_rows: requires at least 2D tensor" << std::endl;
        exit(-1);
    }
    int cols = shape.back();
    int rows = (int)(size() / cols);
    std::vector<int> out_shape = {1, cols};
    Tensor result(out_shape, 0.0f, device);
    if (device == Device::CUDA) {
        cuda_ops::sum_rows(get_data_ptr(), result.get_data_ptr(), rows, cols);
        return result;
    }
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            result.data[c] += data[r * cols + c];
        }
    }
    return result;
}

void Tensor::sgd_step(float lr) {
    if (device == Device::CUDA) {
        cuda_ops::sgd_step(get_data_ptr(), get_grad_ptr(), lr, 0.0f, size());
        return;
    }
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= lr * grad[i];
    }
}

void Tensor::adamw_step(Tensor& m, Tensor& v, float lr, float beta1, float beta2, float eps, float weight_decay, int t) {
    if (device == Device::CUDA) {
        cuda_ops::adamw_step(get_data_ptr(), get_grad_ptr(), m.get_data_ptr(), v.get_data_ptr(), lr, beta1, beta2, eps, weight_decay, t, size());
        return;
    }
    float m_hat_scale = 1.0f / (1.0f - std::pow(beta1, (float)t));
    float v_hat_scale = 1.0f / (1.0f - std::pow(beta2, (float)t));
    for (size_t i = 0; i < data.size(); ++i) {
        float g = grad[i];
        m.data[i] = beta1 * m.data[i] + (1.0f - beta1) * g;
        v.data[i] = beta2 * v.data[i] + (1.0f - beta2) * g * g;
        float m_hat = m.data[i] * m_hat_scale;
        float v_hat = v.data[i] * v_hat_scale;
        data[i] -= lr * weight_decay * data[i];
        data[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

Tensor Tensor::layer_norm(int channels, const Tensor& scale, const Tensor& shift, float eps, Tensor& out_mean, Tensor& out_var, Tensor& out_x_hat) const {
    std::vector<int> mean_var_shape = shape;
    mean_var_shape.back() = 1;
    out_mean = Tensor(mean_var_shape, 0.0f, device);
    out_var = Tensor(mean_var_shape, 0.0f, device);
    out_x_hat = Tensor(shape, 0.0f, device);
    Tensor output(shape, 0.0f, device);

    int total_tokens = (int)(size() / channels);

    if (device == Device::CUDA) {
        cuda_ops::layer_norm(get_data_ptr(), scale.get_data_ptr(), shift.get_data_ptr(), eps, output.get_data_ptr(), out_mean.get_data_ptr(), out_var.get_data_ptr(), out_x_hat.get_data_ptr(), total_tokens, channels);
        return output;
    }

    for (int i = 0; i < total_tokens; ++i) {
        const float* x_ptr = data.data() + i * channels;
        float* x_hat_ptr = out_x_hat.data.data() + i * channels;
        float* out_ptr = output.data.data() + i * channels;

        float sum = 0.0f;
        for (int j = 0; j < channels; ++j) {
            sum += x_ptr[j];
        }
        float mean = sum / (float)channels;
        out_mean.data[i] = mean;

        float var_sum = 0.0f;
        for (int j = 0; j < channels; ++j) {
            float diff = x_ptr[j] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / (float)channels;
        out_var.data[i] = var;

        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int j = 0; j < channels; ++j) {
            float x_hat = (x_ptr[j] - mean) * inv_std;
            x_hat_ptr[j] = x_hat;
            out_ptr[j] = scale.data[j] * x_hat + shift.data[j];
        }
    }
    return output;
}

void Tensor::layer_norm_into(int channels, const Tensor& scale, const Tensor& shift, float eps, Tensor& out_mean, Tensor& out_var, Tensor& out_x_hat, Tensor& output) const {
    std::vector<int> mean_var_shape = shape;
    mean_var_shape.back() = 1;
    if (out_mean.shape != mean_var_shape || out_mean.device != device || (!out_mean.cuda_data && device == Device::CUDA)) {
        out_mean = Tensor(mean_var_shape, 0.0f, device);
    }
    if (out_var.shape != mean_var_shape || out_var.device != device || (!out_var.cuda_data && device == Device::CUDA)) {
        out_var = Tensor(mean_var_shape, 0.0f, device);
    }
    if (out_x_hat.shape != shape || out_x_hat.device != device || (!out_x_hat.cuda_data && device == Device::CUDA)) {
        out_x_hat = Tensor(shape, 0.0f, device);
    }
    if (output.shape != shape || output.device != device || (!output.cuda_data && device == Device::CUDA)) {
        output = Tensor(shape, 0.0f, device);
    }
    int total_tokens = (int)(size() / channels);
    if (device == Device::CUDA) {
        cuda_ops::layer_norm(get_data_ptr(), scale.get_data_ptr(), shift.get_data_ptr(), eps, output.get_data_ptr(), out_mean.get_data_ptr(), out_var.get_data_ptr(), out_x_hat.get_data_ptr(), total_tokens, channels);
    } else {
        output = layer_norm(channels, scale, shift, eps, out_mean, out_var, out_x_hat);
    }
}

Tensor Tensor::layer_norm_backward(const Tensor& dout, const Tensor& x_hat, Tensor& scale, Tensor& shift, const Tensor& mean, const Tensor& var, float eps) const {
    if (dout.shape != shape) {
        std::cerr << "layer_norm_backward: dout shape mismatch!" << std::endl;
        exit(-1);
    }
    int channels = shape.back();
    int total_tokens = (int)(dout.size() / channels);
    Tensor dX(dout.shape, 0.0f, device);

    if (device == Device::CUDA) {
        cuda_ops::layer_norm_backward(dout.get_data_ptr(), x_hat.get_data_ptr(), scale.get_data_ptr(), scale.get_grad_ptr(), shift.get_grad_ptr(), var.get_data_ptr(), eps, dX.get_data_ptr(), total_tokens, channels);
        return dX;
    }

    for (int i = 0; i < total_tokens; ++i) {
        const float* dout_ptr = dout.data.data() + i * channels;
        const float* x_hat_ptr = x_hat.data.data() + i * channels;
        float* dx_ptr = dX.data.data() + i * channels;

        float v = var.data[i];
        float inv_std = 1.0f / std::sqrt(v + eps);

        float sum_dx_hat = 0.0f;
        float sum_dx_hat_x_hat = 0.0f;

        for (int j = 0; j < channels; ++j) {
            float dy = dout_ptr[j];
            shift.grad[j] += dy;
            scale.grad[j] += dy * x_hat_ptr[j];

            float dx_hat = dy * scale.data[j];
            sum_dx_hat += dx_hat;
            sum_dx_hat_x_hat += dx_hat * x_hat_ptr[j];
        }

        float mean_sum_dx_hat = sum_dx_hat / (float)channels;
        float mean_sum_dx_hat_x_hat = sum_dx_hat_x_hat / (float)channels;

        for (int j = 0; j < channels; ++j) {
            float dx_hat = dout_ptr[j] * scale.data[j];
            dx_ptr[j] = inv_std * (dx_hat - mean_sum_dx_hat - x_hat_ptr[j] * mean_sum_dx_hat_x_hat);
        }
    }
    return dX;
}

Tensor Tensor::softmax(int dim) const {
    if (shape.empty()) return Tensor();
    int V = shape.back();
    int total_tokens = (int)(size() / V);
    Tensor probs(shape, 0.0f, device);

    if (device == Device::CUDA) {
        cuda_ops::softmax(get_data_ptr(), probs.get_data_ptr(), total_tokens, V);
        return probs;
    }

    for (int i = 0; i < total_tokens; ++i) {
        const float* logit_ptr = data.data() + i * V;
        float* prob_ptr = probs.data.data() + i * V;

        float max_logit = logit_ptr[0];
        for (int v = 1; v < V; ++v) {
            if (logit_ptr[v] > max_logit) max_logit = logit_ptr[v];
        }

        float sum_exp = 0.0f;
        for (int v = 0; v < V; ++v) {
            float e = std::exp(logit_ptr[v] - max_logit);
            prob_ptr[v] = e;
            sum_exp += e;
        }

        float inv_sum = 1.0f / (sum_exp + 1e-15f);
        for (int v = 0; v < V; ++v) {
            prob_ptr[v] *= inv_sum;
        }
    }
    return probs;
}

Tensor Tensor::softmax_backward(const Tensor& dout) const {
    int V = shape.back();
    int total_tokens = (int)(size() / V);
    Tensor dS(shape, 0.0f, device);

    if (device == Device::CUDA) {
        cuda_ops::softmax_backward(dout.get_data_ptr(), get_data_ptr(), dS.get_data_ptr(), total_tokens, V);
        return dS;
    }

    for (int i = 0; i < total_tokens; ++i) {
        const float* p_ptr = data.data() + i * V;
        const float* dp_ptr = dout.data.data() + i * V;
        float* ds_ptr = dS.data.data() + i * V;

        float sum_dot = 0.0f;
        for (int k = 0; k < V; ++k) {
            sum_dot += dp_ptr[k] * p_ptr[k];
        }
        for (int k = 0; k < V; ++k) {
            ds_ptr[k] = p_ptr[k] * (dp_ptr[k] - sum_dot);
        }
    }
    return dS;
}

float Tensor::cross_entropy_loss(const std::vector<int>& targets, Tensor& out_probs) const {
    out_probs = softmax(-1);
    int V = shape.back();
    int total_tokens = (int)(size() / V);

    if (device == Device::CUDA) {
        static int* d_targets = nullptr;
        static int d_targets_size = 0;
        if (total_tokens > d_targets_size) {
            if (d_targets) cuda_ops::free_memory((float*)d_targets);
            cuda_ops::allocate_memory((float**)&d_targets, total_tokens * sizeof(int) / sizeof(float) + 1);
            d_targets_size = total_tokens;
        }
        cuda_ops::copy_host_to_device((float*)d_targets, (const float*)targets.data(), total_tokens * sizeof(int) / sizeof(float));
        float loss = cuda_ops::cross_entropy_loss(d_targets, out_probs.get_data_ptr(), total_tokens, V);
        return loss;
    }

    float total_loss = 0.0f;
    for (int i = 0; i < total_tokens; ++i) {
        int target_class = targets[i];
        if (target_class < 0 || target_class >= V) {
            std::cerr << "cross_entropy_loss: target class out of bounds: " << target_class << std::endl;
            exit(-1);
        }
        float p = out_probs.data[i * V + target_class];
        total_loss -= std::log(p + 1e-15f);
    }
    return total_loss / (float)total_tokens;
}

Tensor Tensor::cross_entropy_backward(const std::vector<int>& targets) const {
    int V = shape.back();
    int total_tokens = (int)(size() / V);
    Tensor dL(shape, 0.0f, device);

    if (device == Device::CUDA) {
        static int* d_targets_bwd = nullptr;
        static int d_targets_bwd_size = 0;
        if (total_tokens > d_targets_bwd_size) {
            if (d_targets_bwd) cuda_ops::free_memory((float*)d_targets_bwd);
            cuda_ops::allocate_memory((float**)&d_targets_bwd, total_tokens * sizeof(int) / sizeof(float) + 1);
            d_targets_bwd_size = total_tokens;
        }
        cuda_ops::copy_host_to_device((float*)d_targets_bwd, (const float*)targets.data(), total_tokens * sizeof(int) / sizeof(float));
        cuda_ops::cross_entropy_backward(d_targets_bwd, get_data_ptr(), dL.get_data_ptr(), total_tokens, V);
        return dL;
    }

    float scale = 1.0f / (float)total_tokens;
    for (int i = 0; i < total_tokens; ++i) {
        int target_class = targets[i];
        const float* prob_ptr = data.data() + i * V;
        float* dl_ptr = dL.data.data() + i * V;

        for (int v = 0; v < V; ++v) {
            float p = prob_ptr[v];
            if (v == target_class) {
                dl_ptr[v] = (p - 1.0f) * scale;
            } else {
                dl_ptr[v] = p * scale;
            }
        }
    }
    return dL;
}

void Tensor::causal_mask() {
    if (shape.size() < 2) return;
    int cols = shape.back();
    int rows = shape[shape.size() - 2];
    int batches = (int)(size() / (rows * cols));

    if (device == Device::CUDA) {
        cuda_ops::causal_mask(get_data_ptr(), batches, rows);
        return;
    }

    for (int b = 0; b < batches; ++b) {
        for (int r = 0; r < rows; ++r) {
            for (int c = r + 1; c < cols; ++c) {
                data[b * (rows * cols) + r * cols + c] = -1e15f;
            }
        }
    }
}

Tensor Tensor::embedding_lookup(const Tensor& input_ids) const {
    int table_size = shape[0];
    int embed_dim = shape[1];
    std::vector<int> out_shape = input_ids.shape;
    out_shape.push_back(embed_dim);
    Tensor output(out_shape, 0.0f, device);

    int total_tokens = (int)input_ids.size();
    if (device == Device::CUDA) {
        cuda_ops::embedding_lookup(get_data_ptr(), input_ids.get_data_ptr(), output.get_data_ptr(), total_tokens, embed_dim, table_size);
        return output;
    }

    for (int i = 0; i < total_tokens; ++i) {
        int token_id = (int)input_ids.data[i];
        if (token_id < 0 || token_id >= table_size) {
            std::cerr << "embedding_lookup: token_id " << token_id << " out of bounds!" << std::endl;
            exit(-1);
        }
        const float* table_row = data.data() + (token_id * embed_dim);
        float* out_row = output.data.data() + (i * embed_dim);
        for (int j = 0; j < embed_dim; ++j) {
            out_row[j] = table_row[j];
        }
    }
    return output;
}

void Tensor::embedding_backward(const Tensor& dout, const Tensor& input_ids) {
    int table_size = shape[0];
    int embed_dim = shape[1];
    int total_tokens = (int)input_ids.size();

    if (device == Device::CUDA) {
        cuda_ops::embedding_backward(dout.get_data_ptr(), input_ids.get_data_ptr(), get_grad_ptr(), total_tokens, embed_dim, table_size);
        return;
    }

    for (int i = 0; i < total_tokens; ++i) {
        int token_id = (int)input_ids.data[i];
        if (token_id < 0 || token_id >= table_size) {
            std::cerr << "embedding_backward: token_id " << token_id << " out of bounds!" << std::endl;
            exit(-1);
        }
        const float* dout_row = dout.data.data() + (i * embed_dim);
        float* grad_row = grad.data() + (token_id * embed_dim);
        for (int j = 0; j < embed_dim; ++j) {
            grad_row[j] += dout_row[j];
        }
    }
}

Tensor Tensor::gelu() const {
    if (device == Device::CUDA) {
        Tensor result(shape, 0.0f, Device::CUDA);
        cuda_ops::map_op(get_data_ptr(), result.get_data_ptr(), size(), 0);
        return result;
    }
    return map([](float x) {
        float x3 = x * x * x;
        float s = 0.7978845608f * (x + 0.044715f * x3);
        return 0.5f * x * (1.0f + std::tanh(s));
    });
}

void Tensor::gelu_into(Tensor& result) const {
    if (result.shape != shape || result.device != device || (!result.cuda_data && device == Device::CUDA)) {
        result = Tensor(shape, 0.0f, device);
    }
    if (device == Device::CUDA) {
        cuda_ops::map_op(get_data_ptr(), result.get_data_ptr(), size(), 0);
    } else {
        result = gelu();
    }
}

Tensor Tensor::gelu_backward(const Tensor& dout) const {
    if (device == Device::CUDA) {
        Tensor d_gelu(shape, 0.0f, Device::CUDA);
        cuda_ops::map_op(get_data_ptr(), d_gelu.get_data_ptr(), size(), 1);
        return dout * d_gelu;
    }
    Tensor d_gelu = map([](float x) {
        float x2 = x * x;
        float x3 = x2 * x;
        float s = 0.7978845608f * (x + 0.044715f * x3);
        float tanh_s = std::tanh(s);
        float sech_s = 1.0f / std::cosh(s);
        float sech2_s = sech_s * sech_s;

        float term1 = 0.5f * (1.0f + tanh_s);
        float term2 = 0.5f * x * sech2_s * 0.7978845608f * (1.0f + 3.0f * 0.044715f * x2);
        return term1 + term2;
    });
    return dout * d_gelu;
}

Tensor Tensor::concat_channels(const std::vector<Tensor>& head_tensors) {
    if (head_tensors.empty()) {
        std::cerr << "concat_channels: head_tensors is empty" << std::endl;
        exit(-1);
    }
    int num_heads = (int)head_tensors.size();
    std::vector<int> out_shape = head_tensors[0].shape;
    int head_dim = out_shape.back();
    int channels = head_dim * num_heads;
    out_shape.back() = channels;

    int total_tokens = (int)(head_tensors[0].size() / head_dim);
    Device dev = head_tensors[0].device;
    Tensor result(out_shape, 0.0f, dev);

    if (dev == Device::CUDA) {
        for (int h = 0; h < num_heads; ++h) {
            cuda_ops::concat_head(head_tensors[h].get_data_ptr(), result.get_data_ptr(), total_tokens, head_dim, channels, h * head_dim);
        }
    } else {
        for (int tok = 0; tok < total_tokens; ++tok) {
            for (int h = 0; h < num_heads; ++h) {
                const float* src = head_tensors[h].data.data() + tok * head_dim;
                float* dst = result.data.data() + tok * channels + h * head_dim;
                std::copy(src, src + head_dim, dst);
            }
        }
    }
    return result;
}

std::vector<Tensor> Tensor::split_channels(const Tensor& tensor, int num_heads) {
    if (num_heads <= 0 || tensor.shape.empty()) {
        std::cerr << "split_channels: invalid arguments" << std::endl;
        exit(-1);
    }
    int channels = tensor.shape.back();
    if (channels % num_heads != 0) {
        std::cerr << "split_channels: channels not divisible by num_heads" << std::endl;
        exit(-1);
    }
    int head_dim = channels / num_heads;
    std::vector<int> head_shape = tensor.shape;
    head_shape.back() = head_dim;

    int total_tokens = (int)(tensor.size() / channels);
    Device dev = tensor.device;
    std::vector<Tensor> results;
    results.reserve(num_heads);

    for (int h = 0; h < num_heads; ++h) {
        Tensor head(head_shape, 0.0f, dev);
        if (dev == Device::CUDA) {
            cuda_ops::split_head(tensor.get_data_ptr(), head.get_data_ptr(), total_tokens, head_dim, channels, h * head_dim);
        } else {
            for (int tok = 0; tok < total_tokens; ++tok) {
                const float* src = tensor.data.data() + tok * channels + h * head_dim;
                float* dst = head.data.data() + tok * head_dim;
                std::copy(src, src + head_dim, dst);
            }
        }
        results.push_back(std::move(head));
    }
    return results;
}

std::vector<Tensor> Tensor::slice_qkv(const Tensor& qkv, int channels) {
    std::vector<int> out_shape = qkv.shape;
    out_shape.back() = channels;
    int total_tokens = (int)(qkv.size() / (3 * channels));
    Device dev = qkv.device;

    Tensor q(out_shape, 0.0f, dev);
    Tensor k(out_shape, 0.0f, dev);
    Tensor v(out_shape, 0.0f, dev);

    if (dev == Device::CUDA) {
        cuda_ops::slice_qkv(qkv.get_data_ptr(), q.get_data_ptr(), k.get_data_ptr(), v.get_data_ptr(), total_tokens, channels);
    } else {
        for (int tok = 0; tok < total_tokens; ++tok) {
            for (int c = 0; c < channels; ++c) {
                q.data[tok * channels + c] = qkv.data[tok * 3 * channels + c];
                k.data[tok * channels + c] = qkv.data[tok * 3 * channels + channels + c];
                v.data[tok * channels + c] = qkv.data[tok * 3 * channels + 2 * channels + c];
            }
        }
    }
    std::vector<Tensor> results;
    results.reserve(3);
    results.push_back(std::move(q));
    results.push_back(std::move(k));
    results.push_back(std::move(v));
    return results;
}

Tensor Tensor::concat_qkv_grad(const Tensor& dq, const Tensor& dk, const Tensor& dv) {
    std::vector<int> out_shape = dq.shape;
    int channels = dq.shape.back();
    out_shape.back() = 3 * channels;
    int total_tokens = (int)(dq.size() / channels);
    Device dev = dq.device;

    Tensor dqkv(out_shape, 0.0f, dev);
    if (dev == Device::CUDA) {
        cuda_ops::concat_qkv_grad(dq.get_data_ptr(), dk.get_data_ptr(), dv.get_data_ptr(), dqkv.get_data_ptr(), total_tokens, channels);
    } else {
        for (int tok = 0; tok < total_tokens; ++tok) {
            for (int c = 0; c < channels; ++c) {
                dqkv.data[tok * 3 * channels + c] = dq.data[tok * channels + c];
                dqkv.data[tok * 3 * channels + channels + c] = dk.data[tok * channels + c];
                dqkv.data[tok * 3 * channels + 2 * channels + c] = dv.data[tok * channels + c];
            }
        }
    }
    return dqkv;
}

void Tensor::matmul_into(const Tensor& other, Tensor& result) const {
    if (shape.size() == 2 && other.shape.size() == 2) {
        int M = shape[0];
        int K = shape[1];
        int N = other.shape[1];
        std::vector<int> out_shape = {M, N};
        size_t out_size = (size_t)M * N;
        if (result.size() != out_size || result.device != device || (!result.cuda_data && device == Device::CUDA)) {
            result = Tensor(out_shape, 0.0f, device);
        } else {
            result.shape = out_shape;
        }
        if (device == Device::CUDA) {
            cuda_ops::matmul(get_data_ptr(), other.get_data_ptr(), result.get_data_ptr(), M, K, N, 1);
        } else {
            matmul2d_raw(data.data(), other.data.data(), result.data.data(), M, K, N);
        }
        return;
    }
    if (shape.size() == 3 && other.shape.size() == 2) {
        int B = shape[0];
        int M = shape[1];
        int K = shape[2];
        int N = other.shape[1];
        std::vector<int> out_shape = {B, M, N};
        size_t out_size = (size_t)B * M * N;
        if (result.size() != out_size || result.device != device || (!result.cuda_data && device == Device::CUDA)) {
            result = Tensor(out_shape, 0.0f, device);
        } else {
            result.shape = out_shape;
        }
        if (device == Device::CUDA) {
            cuda_ops::matmul(get_data_ptr(), other.get_data_ptr(), result.get_data_ptr(), B * M, K, N, 1);
        } else {
            matmul2d_raw(data.data(), other.data.data(), result.data.data(), B * M, K, N);
        }
        return;
    }
    if (shape.size() == 3 && other.shape.size() == 3) {
        int num_batches = shape[0];
        int M = shape[1];
        int K = shape[2];
        int N = other.shape[2];
        std::vector<int> out_shape = {num_batches, M, N};
        size_t out_size = (size_t)num_batches * M * N;
        if (result.size() != out_size || result.device != device || (!result.cuda_data && device == Device::CUDA)) {
            result = Tensor(out_shape, 0.0f, device);
        } else {
            result.shape = out_shape;
        }
        if (device == Device::CUDA) {
            cuda_ops::matmul(get_data_ptr(), other.get_data_ptr(), result.get_data_ptr(), M, K, N, num_batches);
        } else {
            for (int b = 0; b < num_batches; ++b) {
                const float* a_ptr = data.data() + b * (M * K);
                const float* b_ptr = other.data.data() + b * (K * N);
                float* c_ptr = result.data.data() + b * (M * N);
                matmul2d_raw(a_ptr, b_ptr, c_ptr, M, K, N);
            }
        }
        return;
    }
    result = matmul(other);
}

void Tensor::transpose_into(int dim1, int dim2, Tensor& result) const {
    std::vector<int> new_shape = shape;
    std::swap(new_shape[dim1], new_shape[dim2]);
    if (result.size() != size() || result.device != device || (!result.cuda_data && device == Device::CUDA)) {
        result = Tensor(new_shape, 0.0f, device);
    } else {
        result.shape = new_shape;
    }
    if (device == Device::CUDA) {
        if (shape.size() == 2) {
            cuda_ops::transpose_2d(get_data_ptr(), result.get_data_ptr(), shape[0], shape[1]);
        } else if (shape.size() == 3) {
            cuda_ops::transpose_3d(get_data_ptr(), result.get_data_ptr(), shape[0], shape[1], shape[2], dim1, dim2);
        }
        return;
    }
    result = transpose(dim1, dim2);
}

void Tensor::softmax_into(int dim, Tensor& result) const {
    if (result.shape != shape || result.device != device || !result.cuda_data) {
        result = Tensor(shape, 0.0f, device);
    }
    if (device == Device::CUDA) {
        int total_rows = (int)(size() / shape.back());
        int cols = shape.back();
        cuda_ops::softmax(get_data_ptr(), result.get_data_ptr(), total_rows, cols);
        return;
    }
    result = softmax(dim);
}

void Tensor::add_broadcast_in_place(const Tensor& other) {
    if (device == Device::CUDA && !shape.empty() && other.size() == shape.back()) {
        int C = shape.back();
        int total_rows = (int)(size() / C);
        cuda_ops::add_broadcast(get_data_ptr(), other.get_data_ptr(), get_data_ptr(), total_rows, C);
    } else {
        *this = (*this) + other;
    }
}

void Tensor::mul_scalar_in_place(float scalar) {
    if (device == Device::CUDA) {
        cuda_ops::mul_scalar(get_data_ptr(), scalar, get_data_ptr(), size());
    } else {
        for (size_t i = 0; i < data.size(); i++) data[i] *= scalar;
    }
}

void Tensor::slice_qkv_into(const Tensor& qkv, int channels, std::vector<Tensor>& results) {
    std::vector<int> out_shape = qkv.shape;
    out_shape.back() = channels;
    int total_tokens = (int)(qkv.size() / (3 * channels));
    Device dev = qkv.device;

    if (results.size() != 3) results.resize(3);
    for (int i = 0; i < 3; ++i) {
        if (results[i].shape != out_shape || results[i].device != dev || !results[i].cuda_data) {
            results[i] = Tensor(out_shape, 0.0f, dev);
        }
    }
    if (dev == Device::CUDA) {
        cuda_ops::slice_qkv(qkv.get_data_ptr(), results[0].get_data_ptr(), results[1].get_data_ptr(), results[2].get_data_ptr(), total_tokens, channels);
    } else {
        results = slice_qkv(qkv, channels);
    }
}

void Tensor::split_channels_into(const Tensor& tensor, int num_heads, std::vector<Tensor>& results) {
    int channels = tensor.shape.back();
    int head_dim = channels / num_heads;
    std::vector<int> head_shape = tensor.shape;
    head_shape.back() = head_dim;
    int total_tokens = (int)(tensor.size() / channels);
    Device dev = tensor.device;

    if ((int)results.size() != num_heads) results.resize(num_heads);
    for (int h = 0; h < num_heads; ++h) {
        if (results[h].shape != head_shape || results[h].device != dev || !results[h].cuda_data) {
            results[h] = Tensor(head_shape, 0.0f, dev);
        }
        if (dev == Device::CUDA) {
            cuda_ops::split_head(tensor.get_data_ptr(), results[h].get_data_ptr(), total_tokens, head_dim, channels, h * head_dim);
        }
    }
    if (dev != Device::CUDA) {
        results = split_channels(tensor, num_heads);
    }
}

void Tensor::concat_channels_into(const std::vector<Tensor>& head_tensors, Tensor& result) {
    int num_heads = (int)head_tensors.size();
    std::vector<int> out_shape = head_tensors[0].shape;
    int head_dim = out_shape.back();
    int channels = head_dim * num_heads;
    out_shape.back() = channels;
    int total_tokens = (int)(head_tensors[0].size() / head_dim);
    Device dev = head_tensors[0].device;

    if (result.shape != out_shape || result.device != dev || !result.cuda_data) {
        result = Tensor(out_shape, 0.0f, dev);
    }
    if (dev == Device::CUDA) {
        for (int h = 0; h < num_heads; ++h) {
            cuda_ops::concat_head(head_tensors[h].get_data_ptr(), result.get_data_ptr(), total_tokens, head_dim, channels, h * head_dim);
        }
    } else {
        result = concat_channels(head_tensors);
    }
}

void Tensor::add_into(const Tensor& A, const Tensor& B, Tensor& result) {
    if (result.shape != A.shape || result.device != A.device || (!result.cuda_data && A.device == Device::CUDA)) {
        result = Tensor(A.shape, 0.0f, A.device);
    }
    if (A.device == Device::CUDA) {
        if (B.shape == A.shape) {
            cuda_ops::add(A.get_data_ptr(), B.get_data_ptr(), result.get_data_ptr(), A.size());
        } else if (!A.shape.empty() && B.size() == A.shape.back()) {
            int C = A.shape.back();
            int total_rows = (int)(A.size() / C);
            cuda_ops::add_broadcast(A.get_data_ptr(), B.get_data_ptr(), result.get_data_ptr(), total_rows, C);
        } else {
            std::cerr << "Tensor::add_into shape mismatch!" << std::endl;
            exit(-1);
        }
    } else {
        for (size_t i = 0; i < A.size(); ++i) result.data[i] = A.data[i] + B.data[i % B.size()];
    }
}

void Tensor::sum_rows_into(Tensor& result) const {
    int cols = shape.back();
    int rows = (int)(size() / cols);
    std::vector<int> out_shape = {1, cols};
    if (result.shape != out_shape || result.device != device || (!result.cuda_data && device == Device::CUDA)) {
        result = Tensor(out_shape, 0.0f, device);
    }
    if (device == Device::CUDA) {
        cuda_ops::sum_rows(get_data_ptr(), result.get_data_ptr(), rows, cols);
    } else {
        std::fill(result.data.begin(), result.data.end(), 0.0f);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                result.data[c] += data[r * cols + c];
            }
        }
    }
}

void Tensor::reshape_into(const std::vector<int>& new_shape, Tensor& result) const {
    size_t new_size = 1;
    for (int dim : new_shape) new_size *= dim;
    if (new_size != size()) {
        std::cerr << "Tensor::reshape_into: Total elements must remain identical!" << std::endl;
        exit(-1);
    }
    if (result.shape != new_shape || result.device != device || (!result.cuda_data && device == Device::CUDA)) {
        result = Tensor(new_shape, 0.0f, device);
    } else {
        result.shape = new_shape;
    }
    if (device == Device::CUDA) {
        if (result.cuda_data != cuda_data) {
            cuda_ops::copy_device_to_device(result.cuda_data, cuda_data, size());
        }
    } else {
        result.data = data;
    }
}

void Tensor::softmax_backward_into(const Tensor& dout, Tensor& result) const {
    if (result.shape != shape || result.device != device || (!result.cuda_data && device == Device::CUDA)) {
        result = Tensor(shape, 0.0f, device);
    }
    int cols = shape.back();
    int total_rows = (int)(size() / cols);
    if (device == Device::CUDA) {
        cuda_ops::softmax_backward(dout.get_data_ptr(), get_data_ptr(), result.get_data_ptr(), total_rows, cols);
    } else {
        result = softmax_backward(dout);
    }
}

void Tensor::layer_norm_backward_into(const Tensor& dout, const Tensor& x_hat, Tensor& scale, Tensor& shift, const Tensor& var, float eps, Tensor& dx) const {
    if (dx.shape != shape || dx.device != device || (!dx.cuda_data && device == Device::CUDA)) {
        dx = Tensor(shape, 0.0f, device);
    }
    int channels = shape.back();
    int total_tokens = (int)(size() / channels);
    if (device == Device::CUDA) {
        cuda_ops::layer_norm_backward(dout.get_data_ptr(), x_hat.get_data_ptr(), scale.get_data_ptr(), scale.get_grad_ptr(), shift.get_grad_ptr(), var.get_data_ptr(), eps, dx.get_data_ptr(), total_tokens, channels);
    } else {
        dx = layer_norm_backward(dout, x_hat, scale, shift, Tensor(), var, eps);
    }
}

void Tensor::gelu_backward_into(const Tensor& dout, Tensor& d_gelu_workspace, Tensor& result) const {
    if (d_gelu_workspace.shape != shape || d_gelu_workspace.device != device || (!d_gelu_workspace.cuda_data && device == Device::CUDA)) {
        d_gelu_workspace = Tensor(shape, 0.0f, device);
    }
    if (result.shape != shape || result.device != device || (!result.cuda_data && device == Device::CUDA)) {
        result = Tensor(shape, 0.0f, device);
    }
    if (device == Device::CUDA) {
        cuda_ops::map_op(get_data_ptr(), d_gelu_workspace.get_data_ptr(), size(), 1);
        cuda_ops::mul(dout.get_data_ptr(), d_gelu_workspace.get_data_ptr(), result.get_data_ptr(), size());
    } else {
        result = gelu_backward(dout);
    }
}

void Tensor::concat_qkv_grad_into(const Tensor& dq, const Tensor& dk, const Tensor& dv, Tensor& result) {
    std::vector<int> out_shape = dq.shape;
    int channels = dq.shape.back();
    out_shape.back() = 3 * channels;
    int total_tokens = (int)(dq.size() / channels);
    Device dev = dq.device;
    if (result.shape != out_shape || result.device != dev || (!result.cuda_data && dev == Device::CUDA)) {
        result = Tensor(out_shape, 0.0f, dev);
    }
    if (dev == Device::CUDA) {
        cuda_ops::concat_qkv_grad(dq.get_data_ptr(), dk.get_data_ptr(), dv.get_data_ptr(), result.get_data_ptr(), total_tokens, channels);
    } else {
        result = concat_qkv_grad(dq, dk, dv);
    }
}

void Tensor::embedding_lookup_into(const Tensor& input_ids, Tensor& output) const {
    int table_size = shape[0];
    int embed_dim = shape[1];
    std::vector<int> out_shape = input_ids.shape;
    out_shape.push_back(embed_dim);
    if (output.shape != out_shape || output.device != device || (!output.cuda_data && device == Device::CUDA)) {
        output = Tensor(out_shape, 0.0f, device);
    }
    int total_tokens = (int)input_ids.size();
    if (device == Device::CUDA) {
        cuda_ops::embedding_lookup(get_data_ptr(), input_ids.get_data_ptr(), output.get_data_ptr(), total_tokens, embed_dim, table_size);
    } else {
        output = embedding_lookup(input_ids);
    }
}

void Tensor::permute_qkv_to_heads(const Tensor& qkv_all, Tensor& q, Tensor& k, Tensor& v, int B, int T, int num_heads, int head_dim) {
    std::vector<int> target_shape = {B * num_heads, T, head_dim};
    if (q.shape != target_shape || q.device != qkv_all.device || (!q.cuda_data && qkv_all.device == Device::CUDA)) {
        q = Tensor(target_shape, 0.0f, qkv_all.device);
        k = Tensor(target_shape, 0.0f, qkv_all.device);
        v = Tensor(target_shape, 0.0f, qkv_all.device);
    }
    if (qkv_all.device == Device::CUDA) {
        cuda_ops::permute_qkv_to_heads(qkv_all.get_data_ptr(), q.get_data_ptr(), k.get_data_ptr(), v.get_data_ptr(), B, T, num_heads, head_dim);
    } else {
        int total = B * num_heads * T * head_dim;
        for (int idx = 0; idx < total; ++idx) {
            int d = idx % head_dim;
            int t = (idx / head_dim) % T;
            int h = (idx / (head_dim * T)) % num_heads;
            int b = idx / (head_dim * T * num_heads);
            int in_base = (b * T + t) * (3 * num_heads * head_dim);
            int stride = num_heads * head_dim;
            q.data[idx] = qkv_all.data[in_base + 0 * stride + h * head_dim + d];
            k.data[idx] = qkv_all.data[in_base + 1 * stride + h * head_dim + d];
            v.data[idx] = qkv_all.data[in_base + 2 * stride + h * head_dim + d];
        }
    }
}

void Tensor::permute_heads_grad_to_qkv(const Tensor& dq, const Tensor& dk, const Tensor& dv, Tensor& dqkv_all, int B, int T, int num_heads, int head_dim) {
    std::vector<int> target_shape = {B, T, 3 * num_heads * head_dim};
    if (dqkv_all.shape != target_shape || dqkv_all.device != dq.device || (!dqkv_all.cuda_data && dq.device == Device::CUDA)) {
        dqkv_all = Tensor(target_shape, 0.0f, dq.device);
    }
    if (dq.device == Device::CUDA) {
        cuda_ops::permute_heads_grad_to_qkv(dq.get_data_ptr(), dk.get_data_ptr(), dv.get_data_ptr(), dqkv_all.get_data_ptr(), B, T, num_heads, head_dim);
    } else {
        int total = B * num_heads * T * head_dim;
        for (int idx = 0; idx < total; ++idx) {
            int d = idx % head_dim;
            int t = (idx / head_dim) % T;
            int h = (idx / (head_dim * T)) % num_heads;
            int b = idx / (head_dim * T * num_heads);
            int out_base = (b * T + t) * (3 * num_heads * head_dim);
            int stride = num_heads * head_dim;
            dqkv_all.data[out_base + 0 * stride + h * head_dim + d] = dq.data[idx];
            dqkv_all.data[out_base + 1 * stride + h * head_dim + d] = dk.data[idx];
            dqkv_all.data[out_base + 2 * stride + h * head_dim + d] = dv.data[idx];
        }
    }
}

void Tensor::permute_heads_to_concat(const Tensor& head_ctx, Tensor& concat_ctx, int B, int T, int num_heads, int head_dim) {
    std::vector<int> target_shape = {B, T, num_heads * head_dim};
    if (concat_ctx.shape != target_shape || concat_ctx.device != head_ctx.device || (!concat_ctx.cuda_data && head_ctx.device == Device::CUDA)) {
        concat_ctx = Tensor(target_shape, 0.0f, head_ctx.device);
    }
    if (head_ctx.device == Device::CUDA) {
        cuda_ops::permute_heads_to_concat(head_ctx.get_data_ptr(), concat_ctx.get_data_ptr(), B, T, num_heads, head_dim);
    } else {
        int total = B * num_heads * T * head_dim;
        for (int idx = 0; idx < total; ++idx) {
            int d = idx % head_dim;
            int t = (idx / head_dim) % T;
            int h = (idx / (head_dim * T)) % num_heads;
            int b = idx / (head_dim * T * num_heads);
            int out_idx = (b * T + t) * (num_heads * head_dim) + h * head_dim + d;
            concat_ctx.data[out_idx] = head_ctx.data[idx];
        }
    }
}

void Tensor::permute_concat_to_heads(const Tensor& concat_ctx, Tensor& head_ctx, int B, int T, int num_heads, int head_dim) {
    std::vector<int> target_shape = {B * num_heads, T, head_dim};
    if (head_ctx.shape != target_shape || head_ctx.device != concat_ctx.device || (!head_ctx.cuda_data && concat_ctx.device == Device::CUDA)) {
        head_ctx = Tensor(target_shape, 0.0f, concat_ctx.device);
    }
    if (concat_ctx.device == Device::CUDA) {
        cuda_ops::permute_concat_to_heads(concat_ctx.get_data_ptr(), head_ctx.get_data_ptr(), B, T, num_heads, head_dim);
    } else {
        int total = B * num_heads * T * head_dim;
        for (int idx = 0; idx < total; ++idx) {
            int d = idx % head_dim;
            int t = (idx / head_dim) % T;
            int h = (idx / (head_dim * T)) % num_heads;
            int b = idx / (head_dim * T * num_heads);
            int in_idx = (b * T + t) * (num_heads * head_dim) + h * head_dim + d;
            head_ctx.data[idx] = concat_ctx.data[in_idx];
        }
    }
}

float Tensor::cross_entropy_loss_into(const Tensor& targets, Tensor& out_probs) const {
    softmax_into(-1, out_probs);
    int V = shape.back();
    int total_tokens = (int)(size() / V);
    if (device == Device::CUDA) {
        return cuda_ops::cross_entropy_loss_float(targets.get_data_ptr(), out_probs.get_data_ptr(), total_tokens, V);
    }
    std::vector<int> h_targets(total_tokens);
    for (int i = 0; i < total_tokens; ++i) h_targets[i] = (int)targets.data[i];
    return cross_entropy_loss(h_targets, out_probs);
}

void Tensor::cross_entropy_backward_into(const Tensor& targets, Tensor& result) const {
    int V = shape.back();
    int total_tokens = (int)(size() / V);
    if (result.shape != shape || result.device != device || (!result.cuda_data && device == Device::CUDA)) {
        result = Tensor(shape, 0.0f, device);
    }
    if (device == Device::CUDA) {
        cuda_ops::cross_entropy_backward_float(targets.get_data_ptr(), get_data_ptr(), result.get_data_ptr(), total_tokens, V);
        return;
    }
    std::vector<int> h_targets(total_tokens);
    for (int i = 0; i < total_tokens; ++i) h_targets[i] = (int)targets.data[i];
    result = cross_entropy_backward(h_targets);
}

void Tensor::apply_rope_inplace(Tensor& Q_or_K, 
                                   const Tensor& cos_table, 
                                   const Tensor& sin_table, 
                                   int B, int num_heads, int T, int head_dim, 
                                   bool forward) 
    {
        assert(Q_or_K.device == cos_table.device && Q_or_K.device == sin_table.device &&
           "RoPE error: Q_or_K, cos_table, and sin_table must all be on the same Device!");

        if (Q_or_K.device == Device::CUDA) {
            cuda_ops::apply_rope_cuda(Q_or_K.cuda_data, cos_table.cuda_data, sin_table.cuda_data,
                                      B, num_heads, T, head_dim, forward);
        } else {
            // matrix is on the CPU.
            for (int b = 0; b < B * num_heads; ++b) {
                for (int t = 0; t < T; ++t) {
                    for (int j = 0; j < head_dim / 2; ++j) {
                        int idx0 = (b * T + t) * head_dim + (2 * j);
                        int idx1 = idx0 + 1;
        
                        float q0 = Q_or_K.data[idx0];
                        float q1 = Q_or_K.data[idx1];
        
                        int table_idx = t * (head_dim / 2) + j;
                        float c = cos_table.data[table_idx];
                        float s = forward ? sin_table.data[table_idx] : -sin_table.data[table_idx];
        
                        Q_or_K.data[idx0] = q0 * c - q1 * s;
                        Q_or_K.data[idx1] = q1 * c + q0 * s;
                    }
                
                }
            }
        }


    }
