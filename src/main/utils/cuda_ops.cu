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
        // 1 Block = 1 Row
        int r = blockIdx.x;
        if (r >= total_rows) return;

        const float* row_in = A + r * cols;
        float* row_out = C + r * cols;

        // Online Softmax variables
        float m_val = -1e20f;
        float d_val = 0.0f;

        // Local thread reduction
        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            float x = row_in[c];
            if (x > m_val) {
                float e = expf(m_val - x);
                d_val = d_val * e + 1.0f;
                m_val = x;
            } else {
                d_val += expf(x - m_val);
            }
        }

        // Shared Memory Reduction across the block
        extern __shared__ float smem[];
        float* s_m = smem;
        float* s_d = smem + blockDim.x;

        s_m[threadIdx.x] = m_val;
        s_d[threadIdx.x] = d_val;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                float m1 = s_m[threadIdx.x];
                float d1 = s_d[threadIdx.x];
                float m2 = s_m[threadIdx.x + stride];
                float d2 = s_d[threadIdx.x + stride];

                float m_new = fmaxf(m1, m2);
                float d_new = d1 * expf(m1 - m_new) + d2 * expf(m2 - m_new);
                
                s_m[threadIdx.x] = m_new;
                s_d[threadIdx.x] = d_new;
            }
            __syncthreads();
        }

        float global_m = s_m[0];
        float global_d = s_d[0];
        float inv_sum = 1.0f / fmaxf(global_d, 1e-9f);

        // Final write pass (coalesced)
        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            float x = row_in[c];
            float val = expf(x - global_m) * inv_sum;
            row_out[c] = fminf(fmaxf(val, 1e-7f), 1.0f);
        }
    }

    void softmax(const float* A, float* C, int total_rows, int cols) {
        if (total_rows == 0 || cols == 0) return;
        int threads = 256;
        int blocks = total_rows;
        size_t shared_mem_bytes = 2 * threads * sizeof(float);
        softmax_kernel<<<blocks, threads, shared_mem_bytes>>>(A, C, total_rows, cols);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void softmax_backward_kernel(const float* dout, const float* probs, float* dS, int total_rows, int cols) {
        int r = blockIdx.x;
        if (r >= total_rows) return;

        const float* dout_row = dout + r * cols;
        const float* probs_row = probs + r * cols;
        float* ds_row = dS + r * cols;

        float local_dot = 0.0f;
        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            local_dot += dout_row[c] * probs_row[c];
        }

        extern __shared__ float s_dot[];
        s_dot[threadIdx.x] = local_dot;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                s_dot[threadIdx.x] += s_dot[threadIdx.x + stride];
            }
            __syncthreads();
        }

        float global_dot = s_dot[0];

        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            ds_row[c] = probs_row[c] * (dout_row[c] - global_dot);
        }
    }

    void softmax_backward(const float* dout, const float* probs, float* dS, int total_rows, int cols) {
        if (total_rows == 0 || cols == 0) return;
        int threads = 256;
        int blocks = total_rows;
        size_t shared_mem_bytes = threads * sizeof(float);
        softmax_backward_kernel<<<blocks, threads, shared_mem_bytes>>>(dout, probs, dS, total_rows, cols);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void layer_norm_kernel(const float* x, const float* scale, const float* shift, float eps, float* out, float* mean, float* var, float* x_hat, int total_tokens, int channels) {
        int tok = blockIdx.x;
        if (tok >= total_tokens) return;

        const float* x_tok = x + tok * channels;
        float* out_tok = out + tok * channels;
        float* x_hat_tok = x_hat + tok * channels;

        extern __shared__ float s_mem[];

        float local_sum = 0.0f;
        for (int c = threadIdx.x; c < channels; c += blockDim.x) {
            local_sum += x_tok[c];
        }
        s_mem[threadIdx.x] = local_sum;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                s_mem[threadIdx.x] += s_mem[threadIdx.x + stride];
            }
            __syncthreads();
        }

        float global_m = s_mem[0] / channels;
        if (threadIdx.x == 0) mean[tok] = global_m;
        __syncthreads();

        float local_var_sum = 0.0f;
        for (int c = threadIdx.x; c < channels; c += blockDim.x) {
            float diff = x_tok[c] - global_m;
            local_var_sum += diff * diff;
        }
        s_mem[threadIdx.x] = local_var_sum;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                s_mem[threadIdx.x] += s_mem[threadIdx.x + stride];
            }
            __syncthreads();
        }

        float global_v = s_mem[0] / channels;
        if (threadIdx.x == 0) var[tok] = global_v;
        float inv_std = 1.0f / sqrtf(global_v + eps);

        for (int c = threadIdx.x; c < channels; c += blockDim.x) {
            float norm = (x_tok[c] - global_m) * inv_std;
            x_hat_tok[c] = norm;
            out_tok[c] = norm * scale[c] + shift[c];
        }
    }

    void layer_norm(const float* x, const float* scale, const float* shift, float eps, float* out, float* mean, float* var, float* x_hat, int total_tokens, int channels) {
        if (total_tokens == 0 || channels == 0) return;
        int threads = 256;
        int blocks = total_tokens;
        size_t shared_mem_bytes = threads * sizeof(float);
        layer_norm_kernel<<<blocks, threads, shared_mem_bytes>>>(x, scale, shift, eps, out, mean, var, x_hat, total_tokens, channels);
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
        int param_blocks = (channels + threads - 1) / threads;
        layer_norm_backward_params_kernel<<<param_blocks, threads>>>(dout, x_hat, scale_grad, shift_grad, total_tokens, channels);
        int blocks = (total_tokens + threads - 1) / threads;
        layer_norm_backward_dx_kernel<<<blocks, threads>>>(dout, x_hat, scale, var, eps, dx, total_tokens, channels);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void rms_norm_kernel(const float* x, const float* scale, float eps, float* out, float* rsqrt, float* x_hat, int total_tokens, int channels) {
        int tok = blockIdx.x;
        if (tok >= total_tokens) return;

        const float* x_tok = x + tok * channels;
        float* out_tok = out + tok * channels;
        float* x_hat_tok = x_hat + tok * channels;

        float ms_sum = 0.0f;
        
        for (int c = threadIdx.x; c < channels; c += blockDim.x) {
            float val = x_tok[c];
            ms_sum += val * val;
        }

        extern __shared__ float s_sum[];
        s_sum[threadIdx.x] = ms_sum;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            }
            __syncthreads();
        }

        float global_sum = s_sum[0];
        float ms = global_sum / channels;
        float r = 1.0f / sqrtf(ms + eps);
        
        if (threadIdx.x == 0) {
            rsqrt[tok] = r;
        }

        for (int c = threadIdx.x; c < channels; c += blockDim.x) {
            float norm = x_tok[c] * r;
            x_hat_tok[c] = norm;
            out_tok[c] = norm * scale[c];
        }
    }

    void rms_norm(const float* x, const float* scale, float eps, float* out, float* rsqrt, float* x_hat, int total_tokens, int channels) {
        if (total_tokens == 0 || channels == 0) return;
        int threads = 256;
        int blocks = total_tokens;
        size_t shared_mem_bytes = threads * sizeof(float);
        rms_norm_kernel<<<blocks, threads, shared_mem_bytes>>>(x, scale, eps, out, rsqrt, x_hat, total_tokens, channels);
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
        int param_blocks = (channels + threads - 1) / threads;
        rms_norm_backward_params_kernel<<<param_blocks, threads>>>(dout, x_hat, scale_grad, total_tokens, channels);
        int blocks = (total_tokens + threads - 1) / threads;
        rms_norm_backward_dx_kernel<<<blocks, threads>>>(dout, x_hat, scale, rsqrt, dx, total_tokens, channels);
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
                float p = fmaxf(probs[tok * vocab_size + target], 1e-7f);
                my_loss = -logf(p);
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

    float sum_squares(const float* d_arr, int N) {
        if (N <= 0 || !d_arr) return 0.0f;
        cublasHandle_t handle = get_cublas_handle();
        float result = 0.0f;
        cublasSdot(handle, N, d_arr, 1, d_arr, 1, &result);
        return result;
    }

    static float* d_global_sq_norm = nullptr;

    void reset_sq_norm() {
        if (!d_global_sq_norm) {
            cudaMalloc(&d_global_sq_norm, sizeof(float));
        }
        cudaMemset(d_global_sq_norm, 0, sizeof(float));
    }

    float get_sq_norm() {
        float result = 0.0f;
        if (d_global_sq_norm) {
            cudaMemcpy(&result, d_global_sq_norm, sizeof(float), cudaMemcpyDeviceToHost);
        }
        return result;
    }

    __global__ void add_sq_norm_kernel(const float* arr, int N, float* d_out) {
        extern __shared__ float sdata[];
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        sdata[tid] = (i < N) ? arr[i] * arr[i] : 0.0f;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicAdd(d_out, sdata[0]);
        }
    }

    void accumulate_sq_norm(const float* d_arr, int N) {
        if (N <= 0 || !d_arr) return;
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        add_sq_norm_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_arr, N, d_global_sq_norm);
        CHECK_CUDA(cudaGetLastError());
    }

    __global__ void scale_inplace_kernel(float* arr, float scale, int N) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            arr[idx] *= scale;
        }
    }

    void scale_inplace(float* d_arr, float scale, int N) {
        if (N <= 0 || !d_arr) return;
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        scale_inplace_kernel<<<blocks, threads>>>(d_arr, scale, N);
        CHECK_CUDA(cudaGetLastError());
    }

    #define FLASH_FWD_TILE 64
    template <int HEAD_DIM>
    __global__ void flash_attention_forward_kernel(
        const float* Q, const float* K, const float* V, float* O, float* L_out, 
        int B, int num_heads, int T) 
    {
        int tx = threadIdx.x; 
        int batch_head_id = blockIdx.y;
        int q_tile_id = blockIdx.x;
        int q_idx = q_tile_id * FLASH_FWD_TILE + tx;

        // Pad by 1 to eliminate the 32-way Bank Conflict on s_Q
        extern __shared__ float s_mem[];
        float* s_Q = s_mem;
        float* s_K = &s_mem[FLASH_FWD_TILE * (HEAD_DIM + 1)];
        float* s_V = &s_mem[2 * FLASH_FWD_TILE * (HEAD_DIM + 1)];

        float m_i = -INFINITY;
        float l_i = 0.0f;
        
        float O_i[HEAD_DIM]; 
        for(int d=0; d<HEAD_DIM; d++) O_i[d] = 0.0f;

        int batch_head_offset = batch_head_id * (T * HEAD_DIM);
        int q_tile_start = q_tile_id * FLASH_FWD_TILE;
        
        #pragma unroll
        for (int i = tx; i < FLASH_FWD_TILE * HEAD_DIM; i += FLASH_FWD_TILE) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            if (q_tile_start + row < T) {
                s_Q[row * (HEAD_DIM + 1) + col] = Q[batch_head_offset + q_tile_start * HEAD_DIM + i];
            }
        }
        __syncthreads();

        float scale = 1.0f / sqrtf((float)HEAD_DIM);
        int num_k_tiles = (T + FLASH_FWD_TILE - 1) / FLASH_FWD_TILE;
        
        for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
            int k_tile_start = k_tile * FLASH_FWD_TILE;
            
            #pragma unroll
            for (int i = tx; i < FLASH_FWD_TILE * HEAD_DIM; i += FLASH_FWD_TILE) {
                int row = i / HEAD_DIM;
                int col = i % HEAD_DIM;
                if (k_tile_start + row < T) {
                    s_K[row * (HEAD_DIM + 1) + col] = K[batch_head_offset + k_tile_start * HEAD_DIM + i];
                    s_V[row * (HEAD_DIM + 1) + col] = V[batch_head_offset + k_tile_start * HEAD_DIM + i];
                }
            }
            __syncthreads();

            if (q_idx < T) {
                int max_k = min(FLASH_FWD_TILE, T - k_tile * FLASH_FWD_TILE);
                for (int k = 0; k < max_k; k++) {
                    int global_k_idx = k_tile * FLASH_FWD_TILE + k;
                    if (global_k_idx > q_idx) continue;

                    float score = 0.0f;
                    for (int d = 0; d < HEAD_DIM; d++) {
                        score += s_Q[tx * (HEAD_DIM + 1) + d] * s_K[k * (HEAD_DIM + 1) + d];
                    }
                    score *= scale;

                    float m_old = m_i;
                    m_i = fmaxf(m_i, score);
                    float correction = expf(m_old - m_i);
                    float exp_score = expf(score - m_i);
                    
                    l_i = l_i * correction + exp_score;
                    
                    for (int d = 0; d < HEAD_DIM; d++) {
                        O_i[d] = O_i[d] * correction + exp_score * s_V[k * (HEAD_DIM + 1) + d];
                    }
                }
            }
            __syncthreads();
        }

        if (q_idx < T) {
            float inv_l = 1.0f / l_i;
            for (int d = 0; d < HEAD_DIM; d++) {
                O[batch_head_offset + q_idx * HEAD_DIM + d] = O_i[d] * inv_l;
            }
            if (L_out) {
                L_out[batch_head_id * T + q_idx] = m_i + logf(l_i);
            }
        }
    }

__global__ void top_k_kernel(const float* input, float* out_values, float* out_indices, int num_tokens, int num_experts, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens) return;
    
    int in_offset = idx * num_experts;
    int out_offset = idx * k;
    
    // Initialize the Top-K output slots in VRAM directly
    for (int i = 0; i < k; i++) {
        out_values[out_offset + i] = -1e9f;
        out_indices[out_offset + i] = -1.0f;
    }
    
    // Loop through ALL experts (Zero restrictions on num_experts)
    for (int e = 0; e < num_experts; e++) {
        float val = input[in_offset + e];
        
        // If this value is better than the worst value in our Top-K list
        if (val > out_values[out_offset + k - 1]) {
            
            // Find its exact rank and shift all smaller values down
            int pos = k - 1;
            while (pos > 0 && val > out_values[out_offset + pos - 1]) {
                out_values[out_offset + pos] = out_values[out_offset + pos - 1];
                out_indices[out_offset + pos] = out_indices[out_offset + pos - 1];
                pos--;
            }
            
            // Insert the new champion
            out_values[out_offset + pos] = val;
            out_indices[out_offset + pos] = (float)e;
        }
    }
}

__global__ void moe_histogram_kernel(const float* top_indices, float* expert_counts, int total_elements, int num_experts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    // Thread reads its one assigned number
    int expert_id = (int)top_indices[idx];
    
    if (expert_id >= 0 && expert_id < num_experts) {
        atomicAdd(&expert_counts[expert_id], 1.0f);
    }
}

__global__ void moe_gather_kernel(const float* input_tokens, const float* top_indices, float* current_offsets, float* gathered_tokens,
  float* sorted_token_ids, int num_tokens, int k, int C) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
        if (idx >= num_tokens * k) return;
    
        int original_token = idx / k;
        
        int expert_id = (int)top_indices[idx];

        int slot = (int)atomicAdd(&current_offsets[expert_id], 1.0f);

        sorted_token_ids[slot] = (float)idx;

        for (int c = 0; c < C; c++) {
            gathered_tokens[slot * C + c] = input_tokens[original_token * C + c];
        }
    }

__global__ void moe_scatter_kernel(const float* gathered_outputs, const float* sorted_token_ids, const float* top_probs, float* final_output, int num_gathered_tokens, int k, int C) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (slot >= num_gathered_tokens) return;
    
    // Look up who this answer belongs to (the map stores the flat index!)
    int original_idx = (int)sorted_token_ids[slot];
    
    // Original token is idx / k
    int original_token = original_idx / k;
    
    // Get the exact probability this choice earned
    float prob = top_probs[original_idx];
    
    // Add the scaled answer back into the final sequence
    for (int c = 0; c < C; c++) {
        float val = gathered_outputs[slot * C + c] * prob;
        atomicAdd(&final_output[original_token * C + c], val);
    }
}

__global__ void moe_scatter_backward_kernel(const float* dOutput, const float* gathered_outputs, const float* sorted_token_ids, const float* top_probs, float* dGathered_outputs, float* d_top_probs, int num_gathered_tokens, int k, int C) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (slot >= num_gathered_tokens) return;
    
    int original_idx = (int)sorted_token_ids[slot];
    int original_token = original_idx / k;
    float prob = top_probs[original_idx];
    
    float dot = 0.0f;
    for (int c = 0; c < C; c++) {
        dGathered_outputs[slot * C + c] = dOutput[original_token * C + c] * prob;
        dot += gathered_outputs[slot * C + c] * dOutput[original_token * C + c];
    }
    d_top_probs[original_idx] = dot;
}

__global__ void moe_gather_backward_kernel(const float* dGathered_inputs, const float* sorted_token_ids, float* dInput, int num_gathered_tokens, int k, int C) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (slot >= num_gathered_tokens) return;
    
    int original_idx = (int)sorted_token_ids[slot];
    int original_token = original_idx / k;
    
    for (int c = 0; c < C; c++) {
        atomicAdd(&dInput[original_token * C + c], dGathered_inputs[slot * C + c]);
    }
}

__global__ void scatter_indices_kernel(const float* src, const float* indices, float* dst, int num_elements, int E, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    int token_id = idx / K;
    int expert_id = (int)indices[idx];
    
    dst[token_id * E + expert_id] = src[idx];
}

    void top_k(const float* input, float* out_values, float* out_indices, int num_tokens, int num_experts, int k) {
        int threads = 256;
        int blocks = (num_tokens + threads - 1) / threads;
        top_k_kernel<<<blocks, threads>>>(input, out_values, out_indices, num_tokens, num_experts, k);
        CHECK_CUDA(cudaGetLastError());
    }

    void moe_histogram(const float* top_indices, float* expert_counts, int num_tokens, int k, int num_experts) {
        int threads = 256;
        int blocks = (num_tokens * k + threads - 1) / threads;
        moe_histogram_kernel<<<blocks,threads>>>(top_indices, expert_counts, num_tokens * k, num_experts);
        CHECK_CUDA(cudaGetLastError());
    }
    
    void moe_gather(const float* input_tokens, const float* top_indices, float* current_offsets, float* gathered_tokens, float*
  sorted_token_ids, int num_tokens, int k, int C) {
            
            int threads = 256; 
            int total_threads_needed = num_tokens * k; 
            
            int blocks = (total_threads_needed + threads - 1) / threads;

            moe_gather_kernel<<<blocks, threads>>>(input_tokens, top_indices, current_offsets, gathered_tokens, sorted_token_ids, num_tokens,
  k, C);
            
            CHECK_CUDA(cudaGetLastError());
        }
    
    void moe_scatter(const float* gathered_outputs, const float* sorted_token_ids, const float* top_probs, float* final_output, int num_gathered_tokens, int k, int C) {
        int threads = 256;
        int blocks = (num_gathered_tokens + threads - 1) / threads;
        moe_scatter_kernel<<<blocks, threads>>>(gathered_outputs, sorted_token_ids, top_probs, final_output, num_gathered_tokens, k, C);
        CHECK_CUDA(cudaGetLastError());
    }
    
    void moe_scatter_backward(const float* dOutput, const float* gathered_outputs, const float* sorted_token_ids, const float* top_probs, float* dGathered_outputs, float* d_top_probs, int num_gathered_tokens, int k, int C) {
        int threads = 256;
        int blocks = (num_gathered_tokens + threads - 1) / threads;
        moe_scatter_backward_kernel<<<blocks, threads>>>(dOutput, gathered_outputs, sorted_token_ids, top_probs, dGathered_outputs, d_top_probs, num_gathered_tokens, k, C);
        CHECK_CUDA(cudaGetLastError());
    }
    
    void moe_gather_backward(const float* dGathered_inputs, const float* sorted_token_ids, float* dInput, int num_gathered_tokens, int k, int C) {
        int threads = 256;
        int blocks = (num_gathered_tokens + threads - 1) / threads;
        moe_gather_backward_kernel<<<blocks, threads>>>(dGathered_inputs, sorted_token_ids, dInput, num_gathered_tokens, k, C);
        CHECK_CUDA(cudaGetLastError());
    }

    void scatter_indices(const float* src, const float* indices, float* dst, int num_elements, int E, int K) {
        int threads = 256;
        int blocks = (num_elements + threads - 1) / threads;
        scatter_indices_kernel<<<blocks, threads>>>(src, indices, dst, num_elements, E, K);
        CHECK_CUDA(cudaGetLastError());
    }

    template <int HEAD_DIM>
    void launch_flash_fwd(const float* Q, const float* K, const float* V, float* O, float* L, int B, int num_heads, int T) {
        dim3 grid((T + FLASH_FWD_TILE - 1) / FLASH_FWD_TILE, B * num_heads);
        dim3 block(FLASH_FWD_TILE);
        size_t shared_mem_size = 3 * FLASH_FWD_TILE * (HEAD_DIM + 1) * sizeof(float);
        
        // Safety Assertion: Ensure the hardware can physically handle the shared memory request
        int max_shared_mem;
        cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        if (shared_mem_size > max_shared_mem) {
            printf("\n[FATAL ERROR] Flash Attention requested %llu bytes of shared memory, but GPU only supports %d bytes!\n", shared_mem_size, max_shared_mem);
            printf("Please increase --heads or decrease --channels to reduce head_dim.\n\n");
            exit(EXIT_FAILURE);
        }

        cudaFuncSetAttribute(flash_attention_forward_kernel<HEAD_DIM>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
        flash_attention_forward_kernel<HEAD_DIM><<<grid, block, shared_mem_size>>>(Q, K, V, O, L, B, num_heads, T);
        CHECK_CUDA(cudaGetLastError());
    }

    void flash_attention_forward(const float* Q, const float* K, const float* V, float* O, float* L, int B, int num_heads, int T, int head_dim) {
        if (head_dim == 64) {
            launch_flash_fwd<64>(Q, K, V, O, L, B, num_heads, T);
        } else if (head_dim == 128) {
            launch_flash_fwd<128>(Q, K, V, O, L, B, num_heads, T);
        } else {
            printf("\n[FATAL ERROR] Unsupported head_dim %d for Flash Attention! Must be exactly 64 or 128.\n", head_dim);
            exit(EXIT_FAILURE);
        }
    }

    #define FLASH_BWD_TILE 64

    __global__ void flash_attention_backward_dq_kernel(
        const float* Q, const float* K, const float* V, const float* O, 
        const float* L, const float* dO, 
        float* dQ, 
        int B, int num_heads, int T, int head_dim) 
    {
        int tx = threadIdx.x; 
        int batch_head_id = blockIdx.y;
        int q_tile_id = blockIdx.x;
        int q_idx = q_tile_id * FLASH_BWD_TILE + tx;

        extern __shared__ float s_mem[];
        float* s_Q = s_mem;
        float* s_O = &s_mem[FLASH_BWD_TILE * head_dim];
        float* s_dO = &s_mem[2 * FLASH_BWD_TILE * head_dim];
        float* s_K = &s_mem[3 * FLASH_BWD_TILE * head_dim];
        float* s_V = &s_mem[4 * FLASH_BWD_TILE * head_dim];

        int batch_head_offset = batch_head_id * (T * head_dim);
        int q_tile_start = q_tile_id * FLASH_BWD_TILE;

        for (int i = tx; i < FLASH_BWD_TILE * head_dim; i += blockDim.x) {
            int row = i / head_dim;
            if (q_tile_start + row < T) {
                s_Q[i] = Q[batch_head_offset + q_tile_start * head_dim + i];
                s_O[i] = O[batch_head_offset + q_tile_start * head_dim + i];
                s_dO[i] = dO[batch_head_offset + q_tile_start * head_dim + i];
            }
        }
        
        float D_i = 0.0f;
        float L_i = 0.0f;
        if (q_idx < T) {
            for(int d=0; d<head_dim; d++) {
                D_i += s_dO[tx * head_dim + d] * s_O[tx * head_dim + d];
            }
            L_i = L[batch_head_id * T + q_idx];
        }
        
        float dQ_i[128]; 
        for(int d=0; d<head_dim; d++) dQ_i[d] = 0.0f;

        __syncthreads();

        float scale = 1.0f / sqrtf((float)head_dim);
        int num_k_tiles = (T + FLASH_BWD_TILE - 1) / FLASH_BWD_TILE;
        
        for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
            int k_tile_start = k_tile * FLASH_BWD_TILE;

            for (int i = tx; i < FLASH_BWD_TILE * head_dim; i += blockDim.x) {
                int row = i / head_dim;
                if (k_tile_start + row < T) {
                    s_K[i] = K[batch_head_offset + k_tile_start * head_dim + i];
                    s_V[i] = V[batch_head_offset + k_tile_start * head_dim + i];
                }
            }
            __syncthreads();

            if (q_idx < T) {
                int max_k = min(FLASH_BWD_TILE, T - k_tile * FLASH_BWD_TILE);
                for (int k = 0; k < max_k; k++) {
                    int global_k_idx = k_tile * FLASH_BWD_TILE + k;
                    if (global_k_idx > q_idx) continue;

                    float score = 0.0f;
                    for(int d=0; d<head_dim; d++) {
                        score += s_Q[tx * head_dim + d] * s_K[k * head_dim + d];
                    }
                    score *= scale;

                    float p = expf(score - L_i);

                    float dP = 0.0f;
                    for(int d=0; d<head_dim; d++) {
                        dP += s_dO[tx * head_dim + d] * s_V[k * head_dim + d];
                    }

                    float dS = p * (dP - D_i) * scale;

                    for(int d=0; d<head_dim; d++) {
                        dQ_i[d] += dS * s_K[k * head_dim + d];
                    }
                }
            }
            __syncthreads();
        }

        if (q_idx < T) {
            for(int d=0; d<head_dim; d++) {
                dQ[batch_head_offset + q_idx * head_dim + d] = dQ_i[d];
            }
        }
    }

    __global__ void flash_attention_backward_dkv_kernel(
        const float* Q, const float* K, const float* V, const float* O, 
        const float* L, const float* dO, 
        float* dK, float* dV, 
        int B, int num_heads, int T, int head_dim) 
    {
        int tx = threadIdx.x; 
        int batch_head_id = blockIdx.y;
        int k_tile_id = blockIdx.x;
        int k_idx = k_tile_id * FLASH_BWD_TILE + tx;

        extern __shared__ float s_mem[];
        float* s_K = s_mem;
        float* s_V = &s_mem[FLASH_BWD_TILE * head_dim];
        float* s_Q = &s_mem[2 * FLASH_BWD_TILE * head_dim];
        float* s_O = &s_mem[3 * FLASH_BWD_TILE * head_dim];
        float* s_dO = &s_mem[4 * FLASH_BWD_TILE * head_dim];

        int batch_head_offset = batch_head_id * (T * head_dim);
        int k_tile_start = k_tile_id * FLASH_BWD_TILE;

        for (int i = tx; i < FLASH_BWD_TILE * head_dim; i += blockDim.x) {
            int row = i / head_dim;
            if (k_tile_start + row < T) {
                s_K[i] = K[batch_head_offset + k_tile_start * head_dim + i];
                s_V[i] = V[batch_head_offset + k_tile_start * head_dim + i];
            }
        }
        
        float dK_i[128]; 
        float dV_i[128];
        for(int d=0; d<head_dim; d++) {
            dK_i[d] = 0.0f;
            dV_i[d] = 0.0f;
        }

        __syncthreads();

        float scale = 1.0f / sqrtf((float)head_dim);
        int num_q_tiles = (T + FLASH_BWD_TILE - 1) / FLASH_BWD_TILE;
        
        for (int q_tile = 0; q_tile < num_q_tiles; q_tile++) {
            int q_tile_start = q_tile * FLASH_BWD_TILE;
            
            if (q_tile_start + FLASH_BWD_TILE - 1 < k_tile_start) continue;

            for (int i = tx; i < FLASH_BWD_TILE * head_dim; i += blockDim.x) {
                int row = i / head_dim;
                if (q_tile_start + row < T) {
                    s_Q[i] = Q[batch_head_offset + q_tile_start * head_dim + i];
                    s_O[i] = O[batch_head_offset + q_tile_start * head_dim + i];
                    s_dO[i] = dO[batch_head_offset + q_tile_start * head_dim + i];
                }
            }
            __syncthreads();

            if (k_idx < T) {
                int max_q = min(FLASH_BWD_TILE, T - q_tile * FLASH_BWD_TILE);
                for (int q = 0; q < max_q; q++) {
                    int global_q_idx = q_tile * FLASH_BWD_TILE + q;
                    if (global_q_idx < k_idx) continue;

                    float D_i = 0.0f;
                    for(int d=0; d<head_dim; d++) {
                        D_i += s_dO[q * head_dim + d] * s_O[q * head_dim + d];
                    }
                    float L_i = L[batch_head_id * T + global_q_idx];

                    float score = 0.0f;
                    for(int d=0; d<head_dim; d++) {
                        score += s_Q[q * head_dim + d] * s_K[tx * head_dim + d];
                    }
                    score *= scale;

                    float p = expf(score - L_i);

                    float dP = 0.0f;
                    for(int d=0; d<head_dim; d++) {
                        dP += s_dO[q * head_dim + d] * s_V[tx * head_dim + d];
                    }

                    float dS = p * (dP - D_i) * scale;

                    for(int d=0; d<head_dim; d++) {
                        dV_i[d] += p * s_dO[q * head_dim + d];
                        dK_i[d] += dS * s_Q[q * head_dim + d];
                    }
                }
            }
            __syncthreads();
        }

        if (k_idx < T) {
            for(int d=0; d<head_dim; d++) {
                dK[batch_head_offset + k_idx * head_dim + d] = dK_i[d];
                dV[batch_head_offset + k_idx * head_dim + d] = dV_i[d];
            }
        }
    }

    void flash_attention_backward(const float* Q, const float* K, const float* V, const float* O, const float* L, const float* dO, float* dQ, float* dK, float* dV, int B, int num_heads, int T, int head_dim) {
        dim3 grid((T + FLASH_BWD_TILE - 1) / FLASH_BWD_TILE, B * num_heads);
        dim3 block(FLASH_BWD_TILE);
        size_t shared_mem_size = 5 * FLASH_BWD_TILE * head_dim * sizeof(float);
        
        cudaFuncSetAttribute(flash_attention_backward_dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
        flash_attention_backward_dq_kernel<<<grid, block, shared_mem_size>>>(Q, K, V, O, L, dO, dQ, B, num_heads, T, head_dim);
        
        cudaFuncSetAttribute(flash_attention_backward_dkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
        flash_attention_backward_dkv_kernel<<<grid, block, shared_mem_size>>>(Q, K, V, O, L, dO, dK, dV, B, num_heads, T, head_dim);
        
        CHECK_CUDA(cudaGetLastError());
    }

} // namespace cuda_ops


__global__ void moe_histogram_kernel(const float* top_indices, float* expert_counts, int total_elements, int num_experts) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (idx >= total_elements) return;
        
        int expert_id = (int)top_indices[idx];
        
        if (expert_id >= 0 && expert_id < num_experts) {
            atomicAdd(&expert_counts[expert_id], 1.0f);
        }
    }

__global__ void top_k_kernel(const float* input, float* out_values, float* out_indices, int num_tokens, int num_experts, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens) return;
    
    int in_offset = idx * num_experts;
    int out_offset = idx * k;
    
    // Initialize the Top-K output slots in VRAM directly
    for (int i = 0; i < k; i++) {
        out_values[out_offset + i] = -1e9f;
        out_indices[out_offset + i] = -1.0f;
    }
    
    for (int e = 0; e < num_experts; e++) {
        float val = input[in_offset + e];
        
        // If this value is better than the worst value in our Top-K list
        if (val > out_values[out_offset + k - 1]) {
            
            // Find its exact rank and shift all smaller values down
            int pos = k - 1;
            while (pos > 0 && val > out_values[out_offset + pos - 1]) {
                out_values[out_offset + pos] = out_values[out_offset + pos - 1];
                out_indices[out_offset + pos] = out_indices[out_offset + pos - 1];
                pos--;
            }
            
            out_values[out_offset + pos] = val;
            out_indices[out_offset + pos] = (float)e;
        }
    }
}
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
    float sum_squares(const float*, int) { return 0.0f; }
    void reset_sq_norm() {}
    void accumulate_sq_norm(const float*, int) {}
    float get_sq_norm() { return 0.0f; }
    void scale_inplace(float*, float, int) {}
    void flash_attention_forward(const float*, const float*, const float*, float*, float*, int, int, int, int) {}
    void flash_attention_backward(const float*, const float*, const float*, const float*, const float*, const float*, float*, float*, float*, int, int, int, int) {}
}

#endif // USE_CUDA
