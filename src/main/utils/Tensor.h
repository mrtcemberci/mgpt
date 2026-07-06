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
    Tensor operator*(float scalar) const;
};



#endif //TENSOR_H
