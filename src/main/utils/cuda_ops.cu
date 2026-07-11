#include "cuda_ops.h"
#include <iostream>
#include <cmath>
#include <cstdlib>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__          \
                      << " -> " << cudaGetErrorString(err) << std::endl;         \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

namespace cuda_ops {

    static int g_malloc_count = 0;
    static int g_free_count = 0;

    void reset_memory_stats() {
        g_malloc_count = 0;
        g_free_count = 0;
    }

    void print_memory_stats(const char* label) {
        std::cout << "      [" << label << "] Mallocs: " << g_malloc_count << " | Frees: " << g_free_count << "\n";
    }

    void allocate_memory(float** ptr, size_t size) {
        if (size == 0) {
            *ptr = nullptr;
            return;
        }
        g_malloc_count++;
        CHECK_CUDA(cudaMalloc((void**)ptr, size * sizeof(float)));
    }

    void free_memory(float* ptr) {
        if (ptr) {
            g_free_count++;
            CHECK_CUDA(cudaFree(ptr));
        }
    }

    void copy_host_to_device(float* dst, const float* src, size_t size) {
        if (size == 0 || !dst || !src) return;
        CHECK_CUDA(cudaMemcpy(dst, src, size * sizeof(float), cudaMemcpyHostToDevice));
    }

    void copy_device_to_host(float* dst, const float* src, size_t size) {
        if (size == 0 || !dst || !src) return;
        CHECK_CUDA(cudaMemcpy(dst, src, size * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void copy_device_to_device(float* dst, const float* src, size_t size) {
        if (size == 0 || !dst || !src) return;
        CHECK_CUDA(cudaMemcpy(dst, src, size * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    void allocate_int_memory(int** ptr, size_t size) {
        if (size == 0) {
            *ptr = nullptr;
            return;
        }
        g_malloc_count++;
        CHECK_CUDA(cudaMalloc((void**)ptr, size * sizeof(int)));
    }

    void free_int_memory(int* ptr) {
        if (ptr) {
            g_free_count++;
            CHECK_CUDA(cudaFree(ptr));
        }
    }

    void copy_int_host_to_device(int* dst, const int* src, size_t size) {
        if (size == 0 || !dst || !src) return;
        CHECK_CUDA(cudaMemcpy(dst, src, size * sizeof(int), cudaMemcpyHostToDevice));
    }

    __global__ void fill_kernel(float* ptr, float val, size_t size) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            ptr[idx] = val;
        }
    }

    void fill(float* ptr, float val, size_t size) {
        if (size == 0 || !ptr) return;
        int threads = 256;
        int blocks = (int)((size + threads - 1) / threads);
        fill_kernel<<<blocks, threads>>>(ptr, val, size);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void add_kernel(const float* A, const float* B, float* C, size_t size) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) C[idx] = A[idx] + B[idx];
    }

    void add(const float* A, const float* B, float* C, size_t size) {
        if (size == 0) return;
        int threads = 256;
        int blocks = (int)((size + threads - 1) / threads);
        add_kernel<<<blocks, threads>>>(A, B, C, size);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void add_scalar_kernel(const float* A, float val, float* C, size_t size) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) C[idx] = A[idx] + val;
    }

    void add_scalar(const float* A, float val, float* C, size_t size) {
        if (size == 0) return;
        int threads = 256;
        int blocks = (int)((size + threads - 1) / threads);
        add_scalar_kernel<<<blocks, threads>>>(A, val, C, size);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void add_broadcast_kernel(const float* A, const float* bias, float* C, int total_rows, int cols) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = total_rows * cols;
        if (idx < total) {
            int col = idx % cols;
            C[idx] = A[idx] + bias[col];
        }
    }

    void add_broadcast(const float* A, const float* bias, float* C, int total_rows, int cols) {
        int total = total_rows * cols;
        if (total == 0) return;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        add_broadcast_kernel<<<blocks, threads>>>(A, bias, C, total_rows, cols);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void sub_kernel(const float* A, const float* B, float* C, size_t size) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) C[idx] = A[idx] - B[idx];
    }

    void sub(const float* A, const float* B, float* C, size_t size) {
        if (size == 0) return;
        int threads = 256;
        int blocks = (int)((size + threads - 1) / threads);
        sub_kernel<<<blocks, threads>>>(A, B, C, size);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void sub_scalar_kernel(const float* A, float val, float* C, size_t size) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) C[idx] = A[idx] - val;
    }

    void sub_scalar(const float* A, float val, float* C, size_t size) {
        if (size == 0) return;
        int threads = 256;
        int blocks = (int)((size + threads - 1) / threads);
        sub_scalar_kernel<<<blocks, threads>>>(A, val, C, size);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void mul_kernel(const float* A, const float* B, float* C, size_t size) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) C[idx] = A[idx] * B[idx];
    }

    void mul(const float* A, const float* B, float* C, size_t size) {
        if (size == 0) return;
        int threads = 256;
        int blocks = (int)((size + threads - 1) / threads);
        mul_kernel<<<blocks, threads>>>(A, B, C, size);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void mul_scalar_kernel(const float* A, float val, float* C, size_t size) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) C[idx] = A[idx] * val;
    }

    void mul_scalar(const float* A, float val, float* C, size_t size) {
        if (size == 0) return;
        int threads = 256;
        int blocks = (int)((size + threads - 1) / threads);
        mul_scalar_kernel<<<blocks, threads>>>(A, val, C, size);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void map_kernel(const float* A, float* C, size_t size, int op_type) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            float x = A[idx];
            if (op_type == 0) { // GELU
                const float SQRT_2_OVER_PI = 0.7978845608028654f;
                float cube = 0.044715f * x * x * x;
                float inner = SQRT_2_OVER_PI * (x + cube);
                C[idx] = 0.5f * x * (1.0f + tanhf(inner));
            } else if (op_type == 1) { // GELU Backward (approximate derivative)
                const float SQRT_2_OVER_PI = 0.7978845608028654f;
                float x3 = x * x * x;
                float inner = SQRT_2_OVER_PI * (x + 0.044715f * x3);
                float t = tanhf(inner);
                float sech2 = 1.0f - t * t;
                float d_inner = SQRT_2_OVER_PI * (1.0f + 3.0f * 0.044715f * x * x);
                C[idx] = 0.5f * (1.0f + t) + 0.5f * x * sech2 * d_inner;
            } else if (op_type == 2) { // ReLU
                C[idx] = x > 0.0f ? x : 0.0f;
            } else if (op_type == 3) { // Sigmoid
                C[idx] = 1.0f / (1.0f + expf(-x));
            } else {
                C[idx] = x;
            }
        }
    }

    void map_op(const float* A, float* C, size_t size, int op_type) {
        if (size == 0) return;
        int threads = 256;
        int blocks = (int)((size + threads - 1) / threads);
        map_kernel<<<blocks, threads>>>(A, C, size, op_type);
        CHECK_CUDA(cudaGetLastError());
    }

    static cublasHandle_t get_cublas_handle() {
        static cublasHandle_t handle = nullptr;
        if (!handle) {
            cublasCreate(&handle);
            cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
        }
        return handle;
    }

    void matmul(const float* A, const float* B, float* C, int M, int K, int N, int batch_size) {
        if (M == 0 || K == 0 || N == 0 || batch_size == 0) return;
        cublasHandle_t handle = get_cublas_handle();
        float alpha = 1.0f;
        float beta = 0.0f;
        if (batch_size == 1) {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                         B, CUDA_R_32F, N,
                         A, CUDA_R_32F, K, &beta,
                         C, CUDA_R_32F, N,
                         CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
            cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                                       B, CUDA_R_32F, N, (long long)(K * N),
                                       A, CUDA_R_32F, K, (long long)(M * K), &beta,
                                       C, CUDA_R_32F, N, (long long)(M * N), batch_size,
                                       CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }

    __global__ void transpose_2d_kernel(const float* A, float* C, int rows, int cols) {
        int r = blockIdx.y * blockDim.y + threadIdx.y;
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (r < rows && c < cols) {
            C[c * rows + r] = A[r * cols + c];
        }
    }

    void transpose_2d(const float* A, float* C, int rows, int cols) {
        if (rows == 0 || cols == 0) return;
        dim3 threads(16, 16);
        dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
        transpose_2d_kernel<<<blocks, threads>>>(A, C, rows, cols);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void transpose_3d_kernel(const float* A, float* C, int B, int T, int channels, int dim0, int dim1) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = B * T * channels;
        if (idx < total) {
            int b = idx / (T * channels);
            int rem = idx % (T * channels);
            int t = rem / channels;
            int c = rem % channels;
            int new_b = b, new_t = t, new_c = c;
            if ((dim0 == 0 && dim1 == 1) || (dim0 == 1 && dim1 == 0)) { new_b = t; new_t = b; }
            else if ((dim0 == 0 && dim1 == 2) || (dim0 == 2 && dim1 == 0)) { new_b = c; new_c = b; }
            else if ((dim0 == 1 && dim1 == 2) || (dim0 == 2 && dim1 == 1)) { new_t = c; new_c = t; }

            // Simplest safe indexing for 3D transpose:
            int out_dim1 = (dim0 == 0 && dim1 == 1) || (dim0 == 1 && dim1 == 0) ? B : ((dim0 == 1 && dim1 == 2) || (dim0 == 2 && dim1 == 1) ? channels : T);
            int out_dim2 = (dim0 == 1 && dim1 == 2) || (dim0 == 2 && dim1 == 1) ? T : channels;
            int out_idx = new_b * (out_dim1 * out_dim2) + new_t * out_dim2 + new_c;
            C[out_idx] = A[idx];
        }
    }

    void transpose_3d(const float* A, float* C, int B, int T, int channels, int dim0, int dim1) {
        int total = B * T * channels;
        if (total == 0) return;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        transpose_3d_kernel<<<blocks, threads>>>(A, C, B, T, channels, dim0, dim1);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void sum_rows_kernel(const float* A, float* C, int rows, int cols) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < cols) {
            float sum = 0.0f;
            for (int r = 0; r < rows; ++r) {
                sum += A[r * cols + c];
            }
            C[c] = sum;
        }
    }

    void sum_rows(const float* A, float* C, int rows, int cols) {
        if (rows == 0 || cols == 0) return;
        int threads = 256;
        int blocks = (cols + threads - 1) / threads;
        sum_rows_kernel<<<blocks, threads>>>(A, C, rows, cols);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void causal_mask_kernel(float* A, int B, int T) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = B * T * T;
        if (idx < total) {
            int rem = idx % (T * T);
            int r = rem / T;
            int c = rem % T;
            if (c > r) {
                A[idx] = -1e9f;
            }
        }
    }

    void causal_mask(float* A, int B, int T) {
        int total = B * T * T;
        if (total == 0) return;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        causal_mask_kernel<<<blocks, threads>>>(A, B, T);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void softmax_kernel(const float* A, float* C, int total_rows, int cols) {
        int r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r < total_rows) {
            const float* row_in = A + r * cols;
            float* row_out = C + r * cols;
            float max_val = -1e20f;
            for (int c = 0; c < cols; ++c) {
                if (row_in[c] > max_val) max_val = row_in[c];
            }
            float sum_exp = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float e = expf(row_in[c] - max_val);
                row_out[c] = e;
                sum_exp += e;
            }
            for (int c = 0; c < cols; ++c) {
                row_out[c] /= (sum_exp + 1e-9f);
            }
        }
    }

    void softmax(const float* A, float* C, int total_rows, int cols) {
        if (total_rows == 0 || cols == 0) return;
        int threads = 256;
        int blocks = (total_rows + threads - 1) / threads;
        softmax_kernel<<<blocks, threads>>>(A, C, total_rows, cols);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void softmax_backward_kernel(const float* dout, const float* probs, float* dS, int total_rows, int cols) {
        int r = blockIdx.x * blockDim.x + threadIdx.x;
        if (r < total_rows) {
            const float* dout_row = dout + r * cols;
            const float* probs_row = probs + r * cols;
            float* ds_row = dS + r * cols;
            float dot = 0.0f;
            for (int c = 0; c < cols; ++c) {
                dot += dout_row[c] * probs_row[c];
            }
            for (int c = 0; c < cols; ++c) {
                ds_row[c] = probs_row[c] * (dout_row[c] - dot);
            }
        }
    }

    void softmax_backward(const float* dout, const float* probs, float* dS, int total_rows, int cols) {
        if (total_rows == 0 || cols == 0) return;
        int threads = 256;
        int blocks = (total_rows + threads - 1) / threads;
        softmax_backward_kernel<<<blocks, threads>>>(dout, probs, dS, total_rows, cols);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void layer_norm_kernel(const float* x, const float* scale, const float* shift, float eps, float* out, float* mean, float* var, float* x_hat, int total_tokens, int channels) {
        int tok = blockIdx.x * blockDim.x + threadIdx.x;
        if (tok < total_tokens) {
            const float* x_tok = x + tok * channels;
            float* out_tok = out + tok * channels;
            float* x_hat_tok = x_hat + tok * channels;

            float sum = 0.0f;
            for (int c = 0; c < channels; ++c) sum += x_tok[c];
            float m = sum / channels;
            mean[tok] = m;

            float var_sum = 0.0f;
            for (int c = 0; c < channels; ++c) {
                float diff = x_tok[c] - m;
                var_sum += diff * diff;
            }
            float v = var_sum / channels;
            var[tok] = v;

            float inv_std = 1.0f / sqrtf(v + eps);
            for (int c = 0; c < channels; ++c) {
                float norm = (x_tok[c] - m) * inv_std;
                x_hat_tok[c] = norm;
                out_tok[c] = norm * scale[c] + shift[c];
            }
        }
    }

    void layer_norm(const float* x, const float* scale, const float* shift, float eps, float* out, float* mean, float* var, float* x_hat, int total_tokens, int channels) {
        if (total_tokens == 0 || channels == 0) return;
        int threads = 256;
        int blocks = (total_tokens + threads - 1) / threads;
        layer_norm_kernel<<<blocks, threads>>>(x, scale, shift, eps, out, mean, var, x_hat, total_tokens, channels);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void layer_norm_backward_dx_kernel(const float* dout, const float* x_hat, const float* scale, const float* var, float eps, float* dx, int total_tokens, int channels) {
        int tok = blockIdx.x * blockDim.x + threadIdx.x;
        if (tok < total_tokens) {
            const float* dout_tok = dout + tok * channels;
            const float* x_hat_tok = x_hat + tok * channels;
            float* dx_tok = dx + tok * channels;
            float inv_std = 1.0f / sqrtf(var[tok] + eps);

            float sum_dx_hat = 0.0f;
            float sum_dx_hat_xhat = 0.0f;
            for (int c = 0; c < channels; ++c) {
                float dx_hat_c = dout_tok[c] * scale[c];
                sum_dx_hat += dx_hat_c;
                sum_dx_hat_xhat += dx_hat_c * x_hat_tok[c];
            }

            for (int c = 0; c < channels; ++c) {
                float dx_hat_c = dout_tok[c] * scale[c];
                dx_tok[c] = inv_std * (dx_hat_c - sum_dx_hat / channels - x_hat_tok[c] * sum_dx_hat_xhat / channels);
            }
        }
    }

    __global__ void layer_norm_backward_params_kernel(const float* dout, const float* x_hat, float* scale_grad, float* shift_grad, int total_tokens, int channels) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < channels) {
            float sg = 0.0f;
            float shg = 0.0f;
            for (int tok = 0; tok < total_tokens; ++tok) {
                float d = dout[tok * channels + c];
                sg += d * x_hat[tok * channels + c];
                shg += d;
            }
            scale_grad[c] += sg;
            shift_grad[c] += shg;
        }
    }

    void layer_norm_backward(const float* dout, const float* x_hat, const float* scale, float* scale_grad, float* shift_grad, const float* var, float eps, float* dx, int total_tokens, int channels) {
        if (total_tokens == 0 || channels == 0) return;
        int threads = 256;
        int blocks = (total_tokens + threads - 1) / threads;
        layer_norm_backward_dx_kernel<<<blocks, threads>>>(dout, x_hat, scale, var, eps, dx, total_tokens, channels);
        int param_blocks = (channels + threads - 1) / threads;
        layer_norm_backward_params_kernel<<<param_blocks, threads>>>(dout, x_hat, scale_grad, shift_grad, total_tokens, channels);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void rms_norm_kernel(const float* x, const float* scale, float eps, float* out, float* rsqrt, float* x_hat, int total_tokens, int channels) {
        int tok = blockIdx.x * blockDim.x + threadIdx.x;
        if (tok < total_tokens) {
            const float* x_tok = x + tok * channels;
            float* out_tok = out + tok * channels;
            float* x_hat_tok = x_hat + tok * channels;

            float ms_sum = 0.0f;
            for (int c = 0; c < channels; ++c) {
                float val = x_tok[c];
                ms_sum += val * val;
            }
            float ms = ms_sum / channels;
            float r = 1.0f / sqrtf(ms + eps);
            rsqrt[tok] = r;

            for (int c = 0; c < channels; ++c) {
                float norm = x_tok[c] * r;
                x_hat_tok[c] = norm;
                out_tok[c] = norm * scale[c];
            }
        }
    }

    void rms_norm(const float* x, const float* scale, float eps, float* out, float* rsqrt, float* x_hat, int total_tokens, int channels) {
        if (total_tokens == 0 || channels == 0) return;
        int threads = 256;
        int blocks = (total_tokens + threads - 1) / threads;
        rms_norm_kernel<<<blocks, threads>>>(x, scale, eps, out, rsqrt, x_hat, total_tokens, channels);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void rms_norm_backward_dx_kernel(const float* dout, const float* x_hat, const float* scale, const float* rsqrt, float* dx, int total_tokens, int channels) {
        int tok = blockIdx.x * blockDim.x + threadIdx.x;
        if (tok < total_tokens) {
            const float* dout_tok = dout + tok * channels;
            const float* x_hat_tok = x_hat + tok * channels;
            float* dx_tok = dx + tok * channels;
            float r = rsqrt[tok];

            float sum_dx_hat_xhat = 0.0f;
            for (int c = 0; c < channels; ++c) {
                float dx_hat_c = dout_tok[c] * scale[c];
                sum_dx_hat_xhat += dx_hat_c * x_hat_tok[c];
            }

            for (int c = 0; c < channels; ++c) {
                float dx_hat_c = dout_tok[c] * scale[c];
                dx_tok[c] = r * (dx_hat_c - x_hat_tok[c] * sum_dx_hat_xhat / channels);
            }
        }
    }

    __global__ void rms_norm_backward_params_kernel(const float* dout, const float* x_hat, float* scale_grad, int total_tokens, int channels) {
        int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c < channels) {
            float sg = 0.0f;
            for (int tok = 0; tok < total_tokens; ++tok) {
                float d = dout[tok * channels + c];
                sg += d * x_hat[tok * channels + c];
            }
            scale_grad[c] += sg;
        }
    }

    void rms_norm_backward(const float* dout, const float* x_hat, const float* scale, float* scale_grad, const float* rsqrt, float* dx, int total_tokens, int channels) {
        if (total_tokens == 0 || channels == 0) return;
        int threads = 256;
        int blocks = (total_tokens + threads - 1) / threads;
        rms_norm_backward_dx_kernel<<<blocks, threads>>>(dout, x_hat, scale, rsqrt, dx, total_tokens, channels);
        int param_blocks = (channels + threads - 1) / threads;
        rms_norm_backward_params_kernel<<<param_blocks, threads>>>(dout, x_hat, scale_grad, total_tokens, channels);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void embedding_lookup_kernel(const float* table, const float* input, float* output, int total_tokens, int embed_dim, int table_size) {
        int tok = blockIdx.x * blockDim.x + threadIdx.x;
        if (tok < total_tokens) {
            int token_id = (int)input[tok];
            if (token_id >= 0 && token_id < table_size) {
                for (int d = 0; d < embed_dim; ++d) {
                    output[tok * embed_dim + d] = table[token_id * embed_dim + d];
                }
            } else {
                for (int d = 0; d < embed_dim; ++d) {
                    output[tok * embed_dim + d] = 0.0f;
                }
            }
        }
    }

    void embedding_lookup(const float* table, const float* input, float* output, int total_tokens, int embed_dim, int table_size) {
        if (total_tokens == 0 || embed_dim == 0) return;
        int threads = 256;
        int blocks = (total_tokens + threads - 1) / threads;
        embedding_lookup_kernel<<<blocks, threads>>>(table, input, output, total_tokens, embed_dim, table_size);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void embedding_backward_kernel(const float* dout, const float* input, float* table_grad, int total_tokens, int embed_dim, int table_size) {
        int tok = blockIdx.x * blockDim.x + threadIdx.x;
        if (tok < total_tokens) {
            int token_id = (int)input[tok];
            if (token_id >= 0 && token_id < table_size) {
                for (int d = 0; d < embed_dim; ++d) {
                    atomicAdd(&table_grad[token_id * embed_dim + d], dout[tok * embed_dim + d]);
                }
            }
        }
    }

    void embedding_backward(const float* dout, const float* input, float* table_grad, int total_tokens, int embed_dim, int table_size) {
        if (total_tokens == 0 || embed_dim == 0) return;
        int threads = 256;
        int blocks = (total_tokens + threads - 1) / threads;
        embedding_backward_kernel<<<blocks, threads>>>(dout, input, table_grad, total_tokens, embed_dim, table_size);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void sgd_step_kernel(float* param, const float* grad, float lr, float weight_decay, size_t size) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            float g = grad[idx];
            if (weight_decay > 0.0f) g += weight_decay * param[idx];
            param[idx] -= lr * g;
        }
    }

    void sgd_step(float* param, const float* grad, float lr, float weight_decay, size_t size) {
        if (size == 0) return;
        int threads = 256;
        int blocks = (int)((size + threads - 1) / threads);
        sgd_step_kernel<<<blocks, threads>>>(param, grad, lr, weight_decay, size);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void adamw_step_kernel(float* param, const float* grad, float* m, float* v, float lr, float beta1, float beta2, float eps, float weight_decay, int t, size_t size) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            float g = grad[idx];
            if (isnan(g) || isinf(g)) return;
            g = fminf(fmaxf(g, -10.0f), 10.0f);
            m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
            v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
            float m_hat = m[idx] / (1.0f - powf(beta1, (float)t));
            float v_hat = v[idx] / (1.0f - powf(beta2, (float)t));

            if (weight_decay > 0.0f) param[idx] -= lr * weight_decay * param[idx];

            param[idx] -= lr * m_hat / (sqrtf(fmaxf(v_hat, 0.0f)) + eps);
        }
    }

    void adamw_step(float* param, const float* grad, float* m, float* v, float lr, float beta1, float beta2, float eps, float weight_decay, int t, size_t size) {
        if (size == 0) return;
        int threads = 256;
        int blocks = (int)((size + threads - 1) / threads);
        adamw_step_kernel<<<blocks, threads>>>(param, grad, m, v, lr, beta1, beta2, eps, weight_decay, t, size);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void cross_entropy_loss_kernel(const int* targets, const float* probs, float* out_sum, int total_tokens, int vocab_size) {
        int tok = blockIdx.x * blockDim.x + threadIdx.x;
        float my_loss = 0.0f;
        if (tok < total_tokens) {
            int target = targets[tok];
            if (target >= 0 && target < vocab_size) {
                float p = probs[tok * vocab_size + target];
                my_loss = -logf(p + 1e-9f);
            }
        }
        __shared__ float s_loss[256];
        s_loss[threadIdx.x] = my_loss;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                s_loss[threadIdx.x] += s_loss[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            atomicAdd(out_sum, s_loss[0]);
        }
    }

    float cross_entropy_loss(const int* targets, const float* probs, int total_tokens, int vocab_size) {
        if (total_tokens == 0 || vocab_size == 0) return 0.0f;
        static float* d_loss_sum = nullptr;
        if (!d_loss_sum) {
            allocate_memory(&d_loss_sum, 1);
        }
        cudaMemset(d_loss_sum, 0, sizeof(float));

        int threads = 256;
        int blocks = (total_tokens + threads - 1) / threads;
        cross_entropy_loss_kernel<<<blocks, threads>>>(targets, probs, d_loss_sum, total_tokens, vocab_size);
        CHECK_CUDA(cudaGetLastError());

        float h_sum = 0.0f;
        copy_device_to_host(&h_sum, d_loss_sum, 1);
        return h_sum / total_tokens;
    }

    __global__ void cross_entropy_backward_kernel(const int* targets, const float* probs, float* dL, int total_tokens, int vocab_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = total_tokens * vocab_size;
        if (idx < total) {
            int tok = idx / vocab_size;
            int col = idx % vocab_size;
            int target = targets[tok];
            float p = probs[idx];
            dL[idx] = (p - (col == target ? 1.0f : 0.0f)) / (float)total_tokens;
        }
    }

    void cross_entropy_backward(const int* targets, const float* probs, float* dL, int total_tokens, int vocab_size) {
        if (total_tokens == 0 || vocab_size == 0) return;
        int total = total_tokens * vocab_size;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        cross_entropy_backward_kernel<<<blocks, threads>>>(targets, probs, dL, total_tokens, vocab_size);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void cross_entropy_loss_kernel_float(const float* targets, const float* probs, float* out_sum, int total_tokens, int vocab_size) {
        int tok = blockIdx.x * blockDim.x + threadIdx.x;
        float my_loss = 0.0f;
        if (tok < total_tokens) {
            int target = (int)targets[tok];
            if (target >= 0 && target < vocab_size) {
                float p = probs[tok * vocab_size + target];
                my_loss = -logf(p + 1e-9f);
            }
        }
        __shared__ float s_loss[256];
        s_loss[threadIdx.x] = my_loss;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                s_loss[threadIdx.x] += s_loss[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            atomicAdd(out_sum, s_loss[0]);
        }
    }

    float cross_entropy_loss_float(const float* targets, const float* probs, int total_tokens, int vocab_size) {
        if (total_tokens == 0 || vocab_size == 0) return 0.0f;
        static float* d_loss_sum = nullptr;
        if (!d_loss_sum) {
            allocate_memory(&d_loss_sum, 1);
        }
        cudaMemset(d_loss_sum, 0, sizeof(float));

        int threads = 256;
        int blocks = (total_tokens + threads - 1) / threads;
        cross_entropy_loss_kernel_float<<<blocks, threads>>>(targets, probs, d_loss_sum, total_tokens, vocab_size);
        CHECK_CUDA(cudaGetLastError());

        float h_sum = 0.0f;
        copy_device_to_host(&h_sum, d_loss_sum, 1);
        return h_sum / total_tokens;
    }

    __global__ void cross_entropy_backward_kernel_float(const float* targets, const float* probs, float* dL, int total_tokens, int vocab_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = total_tokens * vocab_size;
        if (idx < total) {
            int tok = idx / vocab_size;
            int col = idx % vocab_size;
            int target = (int)targets[tok];
            float p = probs[idx];
            dL[idx] = (p - (col == target ? 1.0f : 0.0f)) / (float)total_tokens;
        }
    }

    void cross_entropy_backward_float(const float* targets, const float* probs, float* dL, int total_tokens, int vocab_size) {
        if (total_tokens == 0 || vocab_size == 0) return;
        int total = total_tokens * vocab_size;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        cross_entropy_backward_kernel_float<<<blocks, threads>>>(targets, probs, dL, total_tokens, vocab_size);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void concat_head_kernel(const float* head_data, float* out_data, int total_tokens, int head_dim, int channels, int head_offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = total_tokens * head_dim;
        if (idx < total) {
            int tok = idx / head_dim;
            int d = idx % head_dim;
            out_data[tok * channels + head_offset + d] = head_data[idx];
        }
    }

    void concat_head(const float* head_data, float* out_data, int total_tokens, int head_dim, int channels, int head_offset) {
        if (total_tokens == 0 || head_dim == 0) return;
        int total = total_tokens * head_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        concat_head_kernel<<<blocks, threads>>>(head_data, out_data, total_tokens, head_dim, channels, head_offset);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void split_head_kernel(const float* in_data, float* head_data, int total_tokens, int head_dim, int channels, int head_offset) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = total_tokens * head_dim;
        if (idx < total) {
            int tok = idx / head_dim;
            int d = idx % head_dim;
            head_data[idx] = in_data[tok * channels + head_offset + d];
        }
    }

    void split_head(const float* in_data, float* head_data, int total_tokens, int head_dim, int channels, int head_offset) {
        if (total_tokens == 0 || head_dim == 0) return;
        int total = total_tokens * head_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        split_head_kernel<<<blocks, threads>>>(in_data, head_data, total_tokens, head_dim, channels, head_offset);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void slice_qkv_kernel(const float* qkv, float* q, float* k, float* v, int total_tokens, int channels) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = total_tokens * channels;
        if (idx < total) {
            int tok = idx / channels;
            int c = idx % channels;
            q[idx] = qkv[tok * 3 * channels + c];
            k[idx] = qkv[tok * 3 * channels + channels + c];
            v[idx] = qkv[tok * 3 * channels + 2 * channels + c];
        }
    }

    void slice_qkv(const float* qkv, float* q, float* k, float* v, int total_tokens, int channels) {
        if (total_tokens == 0 || channels == 0) return;
        int total = total_tokens * channels;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        slice_qkv_kernel<<<blocks, threads>>>(qkv, q, k, v, total_tokens, channels);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void concat_qkv_grad_kernel(const float* dq, const float* dk, const float* dv, float* dqkv, int total_tokens, int channels) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = total_tokens * channels;
        if (idx < total) {
            int tok = idx / channels;
            int c = idx % channels;
            dqkv[tok * 3 * channels + c] = dq[idx];
            dqkv[tok * 3 * channels + channels + c] = dk[idx];
            dqkv[tok * 3 * channels + 2 * channels + c] = dv[idx];
        }
    }

    void concat_qkv_grad(const float* dq, const float* dk, const float* dv, float* dqkv, int total_tokens, int channels) {
        if (total_tokens == 0 || channels == 0) return;
        int total = total_tokens * channels;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        concat_qkv_grad_kernel<<<blocks, threads>>>(dq, dk, dv, dqkv, total_tokens, channels);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void permute_qkv_to_heads_kernel(const float* qkv_all, float* q, float* k, float* v, int B, int T, int num_heads, int head_dim) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = B * num_heads * T * head_dim;
        if (idx < total) {
            int d = idx % head_dim;
            int t = (idx / head_dim) % T;
            int h = (idx / (head_dim * T)) % num_heads;
            int b = idx / (head_dim * T * num_heads);
            int in_base = (b * T + t) * (3 * num_heads * head_dim);
            int stride = num_heads * head_dim;
            q[idx] = qkv_all[in_base + 0 * stride + h * head_dim + d];
            k[idx] = qkv_all[in_base + 1 * stride + h * head_dim + d];
            v[idx] = qkv_all[in_base + 2 * stride + h * head_dim + d];
        }
    }

    void permute_qkv_to_heads(const float* qkv_all, float* q, float* k, float* v, int B, int T, int num_heads, int head_dim) {
        if (B == 0 || T == 0 || num_heads == 0 || head_dim == 0) return;
        int total = B * num_heads * T * head_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        permute_qkv_to_heads_kernel<<<blocks, threads>>>(qkv_all, q, k, v, B, T, num_heads, head_dim);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void permute_heads_grad_to_qkv_kernel(const float* dq, const float* dk, const float* dv, float* dqkv_all, int B, int T, int num_heads, int head_dim) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = B * num_heads * T * head_dim;
        if (idx < total) {
            int d = idx % head_dim;
            int t = (idx / head_dim) % T;
            int h = (idx / (head_dim * T)) % num_heads;
            int b = idx / (head_dim * T * num_heads);
            int out_base = (b * T + t) * (3 * num_heads * head_dim);
            int stride = num_heads * head_dim;
            dqkv_all[out_base + 0 * stride + h * head_dim + d] = dq[idx];
            dqkv_all[out_base + 1 * stride + h * head_dim + d] = dk[idx];
            dqkv_all[out_base + 2 * stride + h * head_dim + d] = dv[idx];
        }
    }

    void permute_heads_grad_to_qkv(const float* dq, const float* dk, const float* dv, float* dqkv_all, int B, int T, int num_heads, int head_dim) {
        if (B == 0 || T == 0 || num_heads == 0 || head_dim == 0) return;
        int total = B * num_heads * T * head_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        permute_heads_grad_to_qkv_kernel<<<blocks, threads>>>(dq, dk, dv, dqkv_all, B, T, num_heads, head_dim);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void permute_heads_to_concat_kernel(const float* head_ctx, float* concat_ctx, int B, int T, int num_heads, int head_dim) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = B * num_heads * T * head_dim;
        if (idx < total) {
            int d = idx % head_dim;
            int t = (idx / head_dim) % T;
            int h = (idx / (head_dim * T)) % num_heads;
            int b = idx / (head_dim * T * num_heads);
            int out_idx = (b * T + t) * (num_heads * head_dim) + h * head_dim + d;
            concat_ctx[out_idx] = head_ctx[idx];
        }
    }

    void permute_heads_to_concat(const float* head_ctx, float* concat_ctx, int B, int T, int num_heads, int head_dim) {
        if (B == 0 || T == 0 || num_heads == 0 || head_dim == 0) return;
        int total = B * num_heads * T * head_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        permute_heads_to_concat_kernel<<<blocks, threads>>>(head_ctx, concat_ctx, B, T, num_heads, head_dim);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void permute_concat_to_heads_kernel(const float* concat_ctx, float* head_ctx, int B, int T, int num_heads, int head_dim) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = B * num_heads * T * head_dim;
        if (idx < total) {
            int d = idx % head_dim;
            int t = (idx / head_dim) % T;
            int h = (idx / (head_dim * T)) % num_heads;
            int b = idx / (head_dim * T * num_heads);
            int in_idx = (b * T + t) * (num_heads * head_dim) + h * head_dim + d;
            head_ctx[idx] = concat_ctx[in_idx];
        }
    }

    void permute_concat_to_heads(const float* concat_ctx, float* head_ctx, int B, int T, int num_heads, int head_dim) {
        if (B == 0 || T == 0 || num_heads == 0 || head_dim == 0) return;
        int total = B * num_heads * T * head_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        permute_concat_to_heads_kernel<<<blocks, threads>>>(concat_ctx, head_ctx, B, T, num_heads, head_dim);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void fill_pos_ids_kernel(float* pos_ids, int B, int T) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < B * T) {
            pos_ids[idx] = (float)(idx % T);
        }
    }

    void fill_pos_ids(float* pos_ids, int B, int T) {
        if (B == 0 || T == 0) return;
        int total = B * T;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        fill_pos_ids_kernel<<<blocks, threads>>>(pos_ids, B, T);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void get_batch_gpu_kernel(const int* d_data, int data_size, int batch_size, int max_seq_len, const int* start_indices, float* x_batch, float* y_batch) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = batch_size * max_seq_len;
        if (idx < total) {
            int b = idx / max_seq_len;
            int t = idx % max_seq_len;
            int start_idx = start_indices[b];
            x_batch[idx] = (float)d_data[start_idx + t];
            y_batch[idx] = (float)d_data[start_idx + t + 1];
        }
    }

    void get_batch_gpu(const int* d_data, int data_size, int batch_size, int max_seq_len, const int* start_indices, float* x_batch, float* y_batch) {
        if (batch_size == 0 || max_seq_len == 0 || !d_data) return;
        int total = batch_size * max_seq_len;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        get_batch_gpu_kernel<<<blocks, threads>>>(d_data, data_size, batch_size, max_seq_len, start_indices, x_batch, y_batch);
        CHECK_CUDA(cudaGetLastError());
    }

    void synchronize() {
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    __global__ void apply_rope_cuda_kernel(float* Q_or_K, 
                                float* cos_table, 
                                float* sin_table, 
                                int B, int num_heads, int T, int head_dim, 
                                bool forward) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        int half_dim = head_dim / 2;
        int total_pairs = B * T * num_heads * half_dim;
        
        if (idx >= total_pairs) return;

        int d_pair = idx % half_dim;
        int t = (idx / half_dim) % T;

        int trig_idx = t * half_dim + d_pair;

        float cos_val = cos_table[trig_idx];
        float sin_val = sin_table[trig_idx];

        int mem_offset = idx * 2;
        float q0 = Q_or_K[mem_offset];
        float q1 = Q_or_K[mem_offset + 1];

        float rotated_q0, rotated_q1;
        if (forward) {
            rotated_q0 = q0 * cos_val - q1 * sin_val;
            rotated_q1 = q1 * cos_val + q0 * sin_val;
        } else {
            rotated_q0 = q0 * cos_val + q1 * sin_val;
            rotated_q1 = q1 * cos_val - q0 * sin_val;
        }

        Q_or_K[mem_offset] = rotated_q0;
        Q_or_K[mem_offset + 1] = rotated_q1;
    }

    void apply_rope_cuda(float* Q_or_K, 
                         float* cos_table, 
                         float* sin_table, 
                         int B, int num_heads, int T, int head_dim, 
                         bool forward) {
        int half_dim = head_dim / 2;
        int total_pairs = B * T * num_heads * half_dim;
        if (total_pairs <= 0) return;
        int threads = 256;
        int blocks = (total_pairs + threads - 1) / threads;
        apply_rope_cuda_kernel<<<blocks, threads>>>(Q_or_K, cos_table, sin_table, B, num_heads, T, head_dim, forward);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void pairwise_mult_into_kernel(const float* a, const float* b, float* result, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= N) return;

        result[idx] = a[idx] * b[idx];
    } 

    void pairwise_mult_into(const float* a, const float* b, float* result, int N) {

        int threads = 256; 
        int blocks = (N + threads - 1) / threads;

        pairwise_mult_into_kernel<<<blocks,threads>>>(a,b,result,N);

        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void swish_inplace_kernel(float* a, int N) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= N) return;

        float x = a[idx];

        a[idx] = x / (1.0f + expf(-x));

    }

    void swish_inplace(float* a, int N) {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        swish_inplace_kernel<<<blocks,threads>>>(a,N);

        CHECK_CUDA(cudaGetLastError());

    }

    __global__ void swish_into_kernel(const float* x, float* result, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;
        float xi = x[idx];
        result[idx] = xi / (1.0f + expf(-xi));
    }

    void swish_into(const float* x, float* result, int N) {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        swish_into_kernel<<<blocks, threads>>>(x, result, N);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void swish_backward_into_kernel(const float* x, const float* dout, float* result, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;
        float xi = x[idx];
        float sig = 1.0f / (1.0f + expf(-xi));
        // swish'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        float swish_grad = sig * (1.0f + xi * (1.0f - sig));
        result[idx] = dout[idx] * swish_grad;
    }

    void swish_backward_into(const float* x, const float* dout, float* result, int N) {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        swish_backward_into_kernel<<<blocks, threads>>>(x, dout, result, N);
        CHECK_CUDA(cudaGetLastError());
    }

} // namespace cuda_ops

#else // !USE_CUDA

namespace cuda_ops {
    void synchronize() {}
    void allocate_memory(float**, size_t) {}
    void free_memory(float*) {}
    void copy_host_to_device(float*, const float*, size_t) {}
    void copy_device_to_host(float*, const float*, size_t) {}
    void copy_device_to_device(float*, const float*, size_t) {}
    void fill(float*, float, size_t) {}
    void add(const float*, const float*, float*, size_t) {}
    void add_scalar(const float*, float, float*, size_t) {}
    void add_broadcast(const float*, const float*, float*, int, int) {}
    void sub(const float*, const float*, float*, size_t) {}
    void sub_scalar(const float*, float, float*, size_t) {}
    void mul(const float*, const float*, float*, size_t) {}
    void mul_scalar(const float*, float, float*, size_t) {}
    void map_op(const float*, float*, size_t, int) {}
    void matmul(const float*, const float*, float*, int, int, int, int) {}
    void transpose_2d(const float*, float*, int, int) {}
    void transpose_3d(const float*, float*, int, int, int, int, int) {}
    void sum_rows(const float*, float*, int, int) {}
    void causal_mask(float*, int, int) {}
    void softmax(const float*, float*, int, int) {}
    void softmax_backward(const float*, const float*, float*, int, int) {}
    void layer_norm(const float*, const float*, const float*, float*, float*, float, int, int) {}
    void layer_norm_backward(const float*, const float*, const float*, float*, float*, const float*, float, float*, int, int) {}
    void rms_norm(const float*, const float*, float, float*, float*, float*, int, int) {}
    void rms_norm_backward(const float*, const float*, const float*, float*, const float*, float*, int, int) {}
    void embedding_lookup(const float*, const float*, float*, int, int, int) {}
    void embedding_backward(const float*, const float*, float*, int, int, int) {}
    void sgd_step(float*, const float*, float, float, size_t) {}
    void adamw_step(float*, const float*, float*, float*, float, float, float, float, float, int, size_t) {}
    float cross_entropy_loss(const int*, const float*, int, int) { return 0.0f; }
    void cross_entropy_backward(const int*, const float*, float*, int, int) {}
    void concat_head(const float*, float*, int, int, int, int) {}
    void split_head(const float*, float*, int, int, int, int) {}
    void slice_qkv(const float*, float*, float*, float*, int, int) {}
    void concat_qkv_grad(const float*, const float*, const float*, float*, int, int) {}
    void permute_qkv_to_heads(const float*, float*, float*, float*, int, int, int, int) {}
    void permute_heads_grad_to_qkv(const float*, const float*, const float*, float*, int, int, int, int) {}
    void permute_heads_to_concat(const float*, float*, int, int, int, int) {}
    void permute_concat_to_heads(const float*, float*, int, int, int, int) {}
    void fill_pos_ids(float*, int, int) {}
    void get_batch_gpu(const int*, int, int, int, const int*, float*, float*) {}
    void allocate_int_memory(int**, size_t) {}
    void free_int_memory(int*) {}
    void copy_int_host_to_device(int*, const int*, size_t) {}
    void apply_rope_cuda(float* Q_or_K, 
                                   float* cos_table, 
                                   float* sin_table, 
                                   int B, int num_heads, int T, int head_dim, 
                                   bool forward) {}
    void pairwise_mult_into(const float* a, const float* b, float* result, int N) {}
    void swish_inplace(float* a, int N) {}
    void swish_into(const float* x, float* result, int N) {}
    void swish_backward_into(const float* x, const float* dout, float* result, int N) {}
}

#endif // USE_CUDA
