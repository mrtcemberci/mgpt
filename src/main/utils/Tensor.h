#ifndef TENSOR_H
#define TENSOR_H
#include <functional>
#include <vector>


class Tensor {
private:
    Tensor applyOperation(const Tensor& other, const std::function<float(float, float)>& operation) const;
    void matmul2d_raw(const float* A, const float* B, float* C, int M, int K, int N) const;
public:

    std::vector<float> data;  // Parameter values or activation values
    std::vector<float> grad;  // Accumulated calculus gradients (same size as data)
    std::vector<int> shape;   // Dimensionality, e.g., {8, 256, 384} for {B, T, C}

    Tensor() = default; // Default no-argument constructor
    Tensor(const std::vector<int>& dims, float init_val = 0.0f);
    static Tensor randn(const std::vector<int>& dims, float mean, float stddev);

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

    // Pattern Family 1: Parameter Updates & Gradient Accumulation
    void add_grad(const Tensor& dgrad);
    Tensor sum_rows() const;
    void sgd_step(float lr);
    void adamw_step(std::vector<float>& m, std::vector<float>& v, float lr, float beta1, float beta2, float eps, float weight_decay, int t);

    // Pattern Family 2: Normalization & Statistics
    Tensor layer_norm(int channels, const Tensor& scale, const Tensor& shift, float eps, Tensor& out_mean, Tensor& out_var, Tensor& out_x_hat) const;
    Tensor layer_norm_backward(const Tensor& dout, const Tensor& x_hat, Tensor& scale, Tensor& shift, const Tensor& mean, const Tensor& var, float eps) const;

    // Pattern Family 3: Probability Distributions (Softmax & Cross-Entropy)
    Tensor softmax(int dim = -1) const;
    Tensor softmax_backward(const Tensor& dout) const;
    float cross_entropy_loss(const std::vector<int>& targets, Tensor& out_probs) const;
    Tensor cross_entropy_backward(const std::vector<int>& targets) const;

    // Pattern Family 4: Causal Attention Masking
    void causal_mask();

    // Pattern Family 5: Table Lookups & Indexing
    Tensor embedding_lookup(const Tensor& input_ids) const;
    void embedding_backward(const Tensor& dout, const Tensor& input_ids);

    // Pattern Family 6: Hardware-Accelerated Activations
    Tensor gelu() const;
    Tensor gelu_backward(const Tensor& dout) const;
};



#endif //TENSOR_H
