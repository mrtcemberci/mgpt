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

void Tensor::add_grad(const Tensor& dgrad) {
    if (this->grad.size() != dgrad.data.size()) {
        std::cerr << "add_grad: size mismatch" << std::endl;
        exit(-1);
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
    int rows = (int)(data.size() / cols);
    std::vector<int> out_shape = {1, cols};
    Tensor result(out_shape, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            result.data[c] += data[r * cols + c];
        }
    }
    return result;
}

void Tensor::sgd_step(float lr) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= lr * grad[i];
    }
}

void Tensor::adamw_step(std::vector<float>& m, std::vector<float>& v, float lr, float beta1, float beta2, float eps, float weight_decay, int t) {
    float m_hat_scale = 1.0f / (1.0f - std::pow(beta1, (float)t));
    float v_hat_scale = 1.0f / (1.0f - std::pow(beta2, (float)t));
    for (size_t i = 0; i < data.size(); ++i) {
        float g = grad[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        float m_hat = m[i] * m_hat_scale;
        float v_hat = v[i] * v_hat_scale;
        data[i] -= lr * weight_decay * data[i];
        data[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}

Tensor Tensor::layer_norm(int channels, const Tensor& scale, const Tensor& shift, float eps, Tensor& out_mean, Tensor& out_var, Tensor& out_x_hat) const {
    std::vector<int> mean_var_shape = shape;
    mean_var_shape.back() = 1;
    out_mean = Tensor(mean_var_shape, 0.0f);
    out_var = Tensor(mean_var_shape, 0.0f);
    out_x_hat = Tensor(shape, 0.0f);
    Tensor output(shape, 0.0f);

    int total_tokens = (int)(data.size() / channels);

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

Tensor Tensor::layer_norm_backward(const Tensor& dout, const Tensor& x_hat, Tensor& scale, Tensor& shift, const Tensor& mean, const Tensor& var, float eps) const {
    if (dout.shape != shape) {
        std::cerr << "layer_norm_backward: dout shape mismatch!" << std::endl;
        exit(-1);
    }
    int channels = shape.back();
    int total_tokens = (int)(dout.size() / channels);
    Tensor dX(dout.shape, 0.0f);

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
    int total_tokens = (int)(data.size() / V);
    Tensor probs(shape, 0.0f);

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
    int total_tokens = (int)(data.size() / V);
    Tensor dS(shape, 0.0f);

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
    int total_tokens = (int)(data.size() / V);
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
    int total_tokens = (int)(data.size() / V);
    Tensor dL(shape, 0.0f);
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
    int batches = (int)(data.size() / (rows * cols));

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
    Tensor output(out_shape, 0.0f);

    int total_tokens = (int)input_ids.size();
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
    return map([](float x) {
        float x3 = x * x * x;
        float s = 0.7978845608f * (x + 0.044715f * x3);
        return 0.5f * x * (1.0f + std::tanh(s));
    });
}

Tensor Tensor::gelu_backward(const Tensor& dout) const {
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


