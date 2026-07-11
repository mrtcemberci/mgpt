#ifndef CUDA_OPS_H
#define CUDA_OPS_H

#include <cstddef>
#include <vector>

namespace cuda_ops {
    void reset_memory_stats();
    void print_memory_stats(const char* label);
    void allocate_memory(float** ptr, size_t size);
    void free_memory(float* ptr);
    void copy_host_to_device(float* dst, const float* src, size_t size);
    void copy_device_to_host(float* dst, const float* src, size_t size);
    void copy_device_to_device(float* dst, const float* src, size_t size);

    void fill(float* ptr, float val, size_t size);
    
    void add(const float* A, const float* B, float* C, size_t size);
    void add_scalar(const float* A, float val, float* C, size_t size);
    void add_broadcast(const float* A, const float* bias, float* C, int total_rows, int cols);
    void sub(const float* A, const float* B, float* C, size_t size);
    void sub_scalar(const float* A, float val, float* C, size_t size);
    void mul(const float* A, const float* B, float* C, size_t size);
    void mul_scalar(const float* A, float val, float* C, size_t size);
    
    // op_type: 0: gelu, 1: gelu_backward, 2: relu, 3: sigmoid
    void map_op(const float* A, float* C, size_t size, int op_type);
    
    void matmul(const float* A, const float* B, float* C, int M, int K, int N, int batch_size = 1);
    void transpose_2d(const float* A, float* C, int rows, int cols);
    void transpose_3d(const float* A, float* C, int B, int T, int channels, int dim0, int dim1);
    
    void sum_rows(const float* A, float* C, int rows, int cols);
    void causal_mask(float* A, int B, int T);
    
    void softmax(const float* A, float* C, int total_rows, int cols);
    void softmax_backward(const float* dout, const float* probs, float* dS, int total_rows, int cols);
    
    void layer_norm(const float* x, const float* scale, const float* shift, float eps, float* out, float* mean, float* var, float* x_hat, int total_tokens, int channels);
    void layer_norm_backward(const float* dout, const float* x_hat, const float* scale, float* scale_grad, float* shift_grad, const float* var, float eps, float* dx, int total_tokens, int channels);
    
    void rms_norm(const float* x, const float* scale, float eps, float* out, float* rsqrt, float* x_hat, int total_tokens, int channels);
    void rms_norm_backward(const float* dout, const float* x_hat, const float* scale, float* scale_grad, const float* rsqrt, float* dx, int total_tokens, int channels);
    
    void embedding_lookup(const float* table, const float* input, float* output, int total_tokens, int embed_dim, int table_size);
    void embedding_backward(const float* dout, const float* input, float* table_grad, int total_tokens, int embed_dim, int table_size);
    
    void sgd_step(float* param, const float* grad, float lr, float weight_decay, size_t size);
    void adamw_step(float* param, const float* grad, float* m, float* v, float lr, float beta1, float beta2, float eps, float weight_decay, int t, size_t size);
    
    float cross_entropy_loss(const int* targets, const float* probs, int total_tokens, int vocab_size);
    void cross_entropy_backward(const int* targets, const float* probs, float* dL, int total_tokens, int vocab_size);
    float cross_entropy_loss_float(const float* targets, const float* probs, int total_tokens, int vocab_size);
    void cross_entropy_backward_float(const float* targets, const float* probs, float* dL, int total_tokens, int vocab_size);

    void concat_head(const float* head_data, float* out_data, int total_tokens, int head_dim, int channels, int head_offset);
    void split_head(const float* in_data, float* head_data, int total_tokens, int head_dim, int channels, int head_offset);

    void slice_qkv(const float* qkv, float* q, float* k, float* v, int total_tokens, int channels);
    void concat_qkv_grad(const float* dq, const float* dk, const float* dv, float* dqkv, int total_tokens, int channels);

    void permute_qkv_to_heads(const float* qkv_all, float* q, float* k, float* v, int B, int T, int num_heads, int head_dim);
    void permute_heads_grad_to_qkv(const float* dq, const float* dk, const float* dv, float* dqkv_all, int B, int T, int num_heads, int head_dim);
    void permute_heads_to_concat(const float* head_ctx, float* concat_ctx, int B, int T, int num_heads, int head_dim);
    void permute_concat_to_heads(const float* concat_ctx, float* head_ctx, int B, int T, int num_heads, int head_dim);

    void fill_pos_ids(float* pos_ids, int B, int T);
    void get_batch_gpu(const int* d_data, int data_size, int batch_size, int max_seq_len, const int* start_indices, float* x_batch, float* y_batch);

    void allocate_int_memory(int** ptr, size_t size);
    void free_int_memory(int* ptr);
    void copy_int_host_to_device(int* dst, const int* src, size_t size);
    void synchronize();
    void apply_rope_cuda(float* Q_or_K, 
                                   float* cos_table, 
                                   float* sin_table, 
                                   int B, int num_heads, int T, int head_dim, 
                                   bool forward);
    
    void pairwise_mult_into(const float* a, const float* b, float* result, int N);
    void swish_inplace(float* a, int N);
    void swish_into(const float* x, float* result, int N);
    void swish_backward_into(const float* x, const float* dout, float* result, int N);
}

#endif // CUDA_OPS_H
