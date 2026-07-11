#ifndef TENSOR_H
#define TENSOR_H
#include <functional>
#include <vector>


enum class Device {
    CPU,
    CUDA
};

// Tensor class uses a vector<float> when in CPU mode and float* when in CUDA mode.
class Tensor {
private:
    Tensor applyOperation(const Tensor& other, const std::function<float(float, float)>& operation) const;
    void matmul2d_raw(const float* A, const float* B, float* C, int M, int K, int N) const;
public:
    Device device = Device::CPU;
    bool is_owning = true;    // True if Tensor owns its buffer; false if wrapping a Scratchpad view, used within the destructor
    std::vector<float> data;  // Parameter values or activation values (CPU)
    std::vector<float> grad;  // Accumulated calculus gradients (CPU)
    float* cuda_data = nullptr; // GPU memory pointer when device == Device::CUDA
    float* cuda_grad = nullptr; // GPU grad pointer when device == Device::CUDA
    std::vector<int> shape;   // Dimensionality, e.g., {8, 256, 384} for {B, T, C}

    Tensor() = default; // Default no-argument constructor
    ~Tensor();
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;

    explicit Tensor(const std::vector<int>& dims, float init_val = 0.0f, Device dev = Device::CPU);
    static Tensor randn(const std::vector<int>& dims, float mean, float stddev, Device dev = Device::CPU);
    static Tensor view(const std::vector<int>& dims, float* vram_ptr, Device dev = Device::CUDA);

    Tensor& to(Device target_device);
    float* get_data_ptr();
    const float* get_data_ptr() const;
    float* get_grad_ptr();
    const float* get_grad_ptr() const;
    void copy_from_host(const float* src);
    void copy_to_host(float* dst) const;

    void randomize(float mean, float stddev);      // Fill tensor data from N(mean, stddev^2)
    void zero_grad();                              // Sets all elements in grad to 0.0f
    size_t size() const;                           // Returns total number of elements (product of shape)
    size_t offset(const std::vector<int>& indices) const; // Converts multi-dimensional indices to 1D flat index

    Tensor matmul(const Tensor& other) const;      // 2D/3D Matrix Multiplication
    Tensor transpose(int dim1, int dim2) const;    // Swaps two dimensions
    Tensor reshape(const std::vector<int>& new_shape) const; // Views data under new dimensions

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;

    // Unary element-wise mapping and scalar broadcasting
    Tensor map(const std::function<float(float)>& func) const;
    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;

    // Parameter Updates & Gradient Accumulation
    void add_grad(const Tensor& dgrad);
    Tensor sum_rows() const;
    void sgd_step(float lr);
    void adamw_step(Tensor& m, Tensor& v, float lr, float beta1, float beta2, float eps, float weight_decay, int t);

    // Normalization & Statistics
    Tensor layer_norm(int channels, const Tensor& scale, const Tensor& shift, float eps, Tensor& out_mean, Tensor& out_var, Tensor& out_x_hat) const;
    void layer_norm_into(int channels, const Tensor& scale, const Tensor& shift, float eps, Tensor& out_mean, Tensor& out_var, Tensor& out_x_hat, Tensor& output) const;
    void rms_norm_into(int channels, const Tensor& scale, float eps, Tensor& out_rsqrt, Tensor& out_x_hat, Tensor& output) const;
    Tensor layer_norm_backward(const Tensor& dout, const Tensor& x_hat, Tensor& scale, Tensor& shift, const Tensor& mean, const Tensor& var, float eps) const;

    // Probability Distributions (Softmax & Cross-Entropy)
    Tensor softmax(int dim = -1) const;
    Tensor softmax_backward(const Tensor& dout) const;
    float cross_entropy_loss(const std::vector<int>& targets, Tensor& out_probs) const;
    Tensor cross_entropy_backward(const std::vector<int>& targets) const;

    // Causal Attention Masking
    void causal_mask();

    // Table Lookups & Indexing
    Tensor embedding_lookup(const Tensor& input_ids) const;
    void embedding_backward(const Tensor& dout, const Tensor& input_ids);

    // Hardware-Accelerated Activations
    Tensor gelu() const;
    void gelu_into(Tensor& result) const;
    Tensor gelu_backward(const Tensor& dout) const;

    // Multi-Head Attention Helpers (Channel Concat / Split)
    static Tensor concat_channels(const std::vector<Tensor>& head_tensors);
    static std::vector<Tensor> split_channels(const Tensor& tensor, int num_heads);
    static std::vector<Tensor> slice_qkv(const Tensor& qkv, int channels);
    static Tensor concat_qkv_grad(const Tensor& dq, const Tensor& dk, const Tensor& dv);

    // Workspace pre-allocation and zero-overhead execution helpers
    void matmul_into(const Tensor& other, Tensor& result) const;
    void pairwise_mult_into(const Tensor& b, Tensor& result) const;
    void swish_inplace();
    void swish_into(Tensor& result) const;
    static void swish_backward_into(const Tensor& x, const Tensor& dout, Tensor& result);
    void transpose_into(int dim1, int dim2, Tensor& result) const;
    void softmax_into(int dim, Tensor& result) const;
    void add_broadcast_in_place(const Tensor& other);
    void mul_scalar_in_place(float scalar);
    static void slice_qkv_into(const Tensor& qkv, int channels, std::vector<Tensor>& results);
    static void split_channels_into(const Tensor& tensor, int num_heads, std::vector<Tensor>& results);
    static void concat_channels_into(const std::vector<Tensor>& head_tensors, Tensor& result);
    static void add_into(const Tensor& A, const Tensor& B, Tensor& result);
    void sum_rows_into(Tensor& result) const;
    void reshape_into(const std::vector<int>& new_shape, Tensor& result) const;
    void softmax_backward_into(const Tensor& dout, Tensor& dx) const;
    void layer_norm_backward_into(const Tensor& dout, const Tensor& x_hat, Tensor& scale, Tensor& shift, const Tensor& var, float eps, Tensor& dx) const;
    void rms_norm_backward_into(const Tensor& dout, const Tensor& x_hat, Tensor& scale, const Tensor& rsqrt, Tensor& dx) const;
    void gelu_backward_into(const Tensor& dout, Tensor& d_gelu_workspace, Tensor& result) const;
    static void concat_qkv_grad_into(const Tensor& dq, const Tensor& dk, const Tensor& dv, Tensor& result);
    static void permute_qkv_to_heads(const Tensor& qkv_all, Tensor& q, Tensor& k, Tensor& v, int B, int T, int num_heads, int head_dim);
    static void permute_heads_grad_to_qkv(const Tensor& dq, const Tensor& dk, const Tensor& dv, Tensor& dqkv_all, int B, int T, int num_heads, int head_dim);
    static void permute_heads_to_concat(const Tensor& head_ctx, Tensor& concat_ctx, int B, int T, int num_heads, int head_dim);
    static void permute_concat_to_heads(const Tensor& concat_ctx, Tensor& head_ctx, int B, int T, int num_heads, int head_dim);
    void embedding_lookup_into(const Tensor& input_ids, Tensor& output) const;
    float cross_entropy_loss_into(const Tensor& targets, Tensor& out_probs) const;
    void cross_entropy_backward_into(const Tensor& targets, Tensor& result) const;
    static void apply_rope_inplace(Tensor& Q_or_K, 
                                   const Tensor& cos_table, 
                                   const Tensor& sin_table, 
                                   int B, int num_heads, int T, int head_dim, 
                                   bool forward);
};



#endif //TENSOR_H
