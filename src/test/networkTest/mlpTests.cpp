#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <memory>
#include "Tensor.h"
#include "LinearLayer.h"
#include "GELULayer.h"

// Helper for relative error in finite difference checking (using 3e-2f tolerance for float32 non-linear precision)
bool check_grad_close(float analytic, float numerical, float tol = 3e-2f) {
    float diff = std::abs(analytic - numerical);
    float denom = std::abs(analytic) + std::abs(numerical) + 1e-8f;
    float rel_error = diff / denom;
    if (rel_error >= tol) {
        std::cerr << "Grad check failed! Analytic: " << analytic 
                  << " vs Numerical: " << numerical << " (Rel Error: " << rel_error << ")\n";
        return false;
    }
    return true;
}

// Simple 2-Layer MLP Wrapper for testing
class TinyMLP {
public:
    LinearLayer fc1;
    GELULayer gelu;
    LinearLayer fc2;

    TinyMLP(int in_dim, int hidden_dim, int out_dim)
        : fc1(in_dim, hidden_dim), gelu(), fc2(hidden_dim, out_dim) {}

    Tensor forward(const Tensor& X) {
        Tensor h = fc1.forward(X);
        Tensor a = gelu.forward(h);
        return fc2.forward(a);
    }

    Tensor backward(const Tensor& dY) {
        Tensor da = fc2.backward(dY);
        Tensor dh = gelu.backward(da);
        return fc1.backward(dh);
    }

    void zero_grad() {
        fc1.weights.zero_grad();
        fc1.biases.zero_grad();
        fc2.weights.zero_grad();
        fc2.biases.zero_grad();
    }
};

// Helper: compute Mean Squared Error (MSE) loss against target T
float compute_mse_loss(const Tensor& Y, const Tensor& T) {
    float sum = 0.0f;
    for (size_t i = 0; i < Y.size(); ++i) {
        float diff = Y.data[i] - T.data[i];
        sum += 0.5f * diff * diff;
    }
    return sum / (float)Y.size();
}

// Helper: compute analytic derivative of MSE loss w.r.t Y
Tensor compute_mse_grad(const Tensor& Y, const Tensor& T) {
    Tensor dY(Y.shape, 0.0f);
    float scale = 1.0f / (float)Y.size();
    for (size_t i = 0; i < Y.size(); ++i) {
        dY.data[i] = (Y.data[i] - T.data[i]) * scale;
    }
    return dY;
}

// ============================================================================
// 1. FORWARD PASS SANITY CHECK
// ============================================================================
void test_mlp_forward() {
    std::cout << "Running Test 1: 2-Layer MLP Forward Pass Sanity Check..." << std::endl;
    TinyMLP mlp(4, 8, 3);
    Tensor X({2, 5, 4}, 1.0f); // Batch=2, Time=5, In=4
    for (size_t i = 0; i < X.size(); ++i) X.data[i] = ((float)(i % 7) - 3.0f) * 0.5f;

    Tensor Y = mlp.forward(X);
    assert(Y.shape.size() == 3 && Y.shape[0] == 2 && Y.shape[1] == 5 && Y.shape[2] == 3);
    assert(Y.size() == 30);
    std::cout << "  -> Forward pass output shape {2, 5, 3} verified! ✅\n";
}

// ============================================================================
// 2-6. FINITE DIFFERENCE GRADIENT CHECKS (The Gold Standard of Backprop)
// ============================================================================
void test_finite_difference_gradcheck() {
    std::cout << "Running Test 2-6: Finite Difference Numerical Gradient Checks..." << std::endl;

    // Small dimensions for fast & rigorous numerical checking
    int B = 2, T = 3, C_in = 3, C_hidden = 5, C_out = 2;
    TinyMLP mlp(C_in, C_hidden, C_out);

    // Initialize weights with deterministic non-zero values
    for (size_t i = 0; i < mlp.fc1.weights.size(); ++i) mlp.fc1.weights.data[i] = ((float)(i % 5) - 2.0f) * 0.2f;
    for (size_t i = 0; i < mlp.fc1.biases.size(); ++i)  mlp.fc1.biases.data[i]  = 0.1f * (float)(i + 1);
    for (size_t i = 0; i < mlp.fc2.weights.size(); ++i) mlp.fc2.weights.data[i] = ((float)(i % 7) - 3.0f) * 0.15f;
    for (size_t i = 0; i < mlp.fc2.biases.size(); ++i)  mlp.fc2.biases.data[i]  = -0.05f * (float)(i + 1);

    Tensor X({B, T, C_in}, 0.0f);
    for (size_t i = 0; i < X.size(); ++i) X.data[i] = ((float)(i % 9) - 4.0f) * 0.3f;

    Tensor Target({B, T, C_out}, 0.0f);
    for (size_t i = 0; i < Target.size(); ++i) Target.data[i] = ((float)(i % 3) - 1.0f) * 0.5f;

    // --- STEP 1: Compute Analytic Gradients ---
    mlp.zero_grad();
    Tensor Y = mlp.forward(X);
    Tensor dY = compute_mse_grad(Y, Target);
    Tensor dX_analytic = mlp.backward(dY);

    float eps = 5e-3f;

    // --- Test 2: Layer 2 Weights Gradcheck ---
    std::cout << "  Checking fc2.weights gradients...";
    for (size_t i = 0; i < mlp.fc2.weights.size(); ++i) {
        float orig = mlp.fc2.weights.data[i];
        
        mlp.fc2.weights.data[i] = orig + eps;
        float loss_plus = compute_mse_loss(mlp.forward(X), Target);
        
        mlp.fc2.weights.data[i] = orig - eps;
        float loss_minus = compute_mse_loss(mlp.forward(X), Target);
        
        mlp.fc2.weights.data[i] = orig; // Restore
        float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
        assert(check_grad_close(mlp.fc2.weights.grad[i], num_grad));
    }
    std::cout << " Passed! ✅\n";

    // --- Test 3: Layer 2 Biases Gradcheck ---
    std::cout << "  Checking fc2.biases gradients...";
    for (size_t i = 0; i < mlp.fc2.biases.size(); ++i) {
        float orig = mlp.fc2.biases.data[i];
        mlp.fc2.biases.data[i] = orig + eps;
        float loss_plus = compute_mse_loss(mlp.forward(X), Target);
        mlp.fc2.biases.data[i] = orig - eps;
        float loss_minus = compute_mse_loss(mlp.forward(X), Target);
        mlp.fc2.biases.data[i] = orig;
        float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
        assert(check_grad_close(mlp.fc2.biases.grad[i], num_grad));
    }
    std::cout << " Passed! ✅\n";

    // --- Test 4: Layer 1 Weights Gradcheck (through GELU!) ---
    std::cout << "  Checking fc1.weights gradients (backprop through GELU)...";
    for (size_t i = 0; i < mlp.fc1.weights.size(); ++i) {
        float orig = mlp.fc1.weights.data[i];
        mlp.fc1.weights.data[i] = orig + eps;
        float loss_plus = compute_mse_loss(mlp.forward(X), Target);
        mlp.fc1.weights.data[i] = orig - eps;
        float loss_minus = compute_mse_loss(mlp.forward(X), Target);
        mlp.fc1.weights.data[i] = orig;
        float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
        assert(check_grad_close(mlp.fc1.weights.grad[i], num_grad));
    }
    std::cout << " Passed! ✅\n";

    // --- Test 5: Layer 1 Biases Gradcheck ---
    std::cout << "  Checking fc1.biases gradients...";
    for (size_t i = 0; i < mlp.fc1.biases.size(); ++i) {
        float orig = mlp.fc1.biases.data[i];
        mlp.fc1.biases.data[i] = orig + eps;
        float loss_plus = compute_mse_loss(mlp.forward(X), Target);
        mlp.fc1.biases.data[i] = orig - eps;
        float loss_minus = compute_mse_loss(mlp.forward(X), Target);
        mlp.fc1.biases.data[i] = orig;
        float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
        assert(check_grad_close(mlp.fc1.biases.grad[i], num_grad));
    }
    std::cout << " Passed! ✅\n";

    // --- Test 6: Input Gradients (dX) Gradcheck ---
    std::cout << "  Checking input dX gradients...";
    for (size_t i = 0; i < X.size(); ++i) {
        float orig = X.data[i];
        X.data[i] = orig + eps;
        float loss_plus = compute_mse_loss(mlp.forward(X), Target);
        X.data[i] = orig - eps;
        float loss_minus = compute_mse_loss(mlp.forward(X), Target);
        X.data[i] = orig;
        float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
        assert(check_grad_close(dX_analytic.data[i], num_grad));
    }
    std::cout << " Passed! ✅\n";
}

// ============================================================================
// 7. TRAINING CONVERGENCE TEST
// ============================================================================
void test_training_convergence() {
    std::cout << "Running Test 7: Multi-Step Training Loss Convergence Check..." << std::endl;
    TinyMLP mlp(2, 8, 1);
    
    // Break symmetry so hidden units learn distinct features
    for (size_t i = 0; i < mlp.fc1.weights.size(); ++i) mlp.fc1.weights.data[i] = ((float)(i % 5) - 2.0f) * 0.2f;
    for (size_t i = 0; i < mlp.fc2.weights.size(); ++i) mlp.fc2.weights.data[i] = ((float)(i % 3) - 1.0f) * 0.2f;

    // Dataset: Learn to predict y = x0 + x1
    Tensor X({4, 2}, 0.0f);
    X.data = {1.0f, 2.0f,  // sum = 3.0
             -1.0f, 3.0f,  // sum = 2.0
              0.5f, 0.5f,  // sum = 1.0
             -2.0f,-2.0f}; // sum = -4.0
    Tensor Target({4, 1}, 0.0f);
    Target.data = {3.0f, 2.0f, 1.0f, -4.0f};

    float lr = 0.05f;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int step = 0; step < 200; ++step) {
        mlp.zero_grad();
        Tensor Y = mlp.forward(X);
        float loss = compute_mse_loss(Y, Target);
        if (step == 0) initial_loss = loss;
        if (step == 199) final_loss = loss;

        Tensor dY = compute_mse_grad(Y, Target);
        mlp.backward(dY);

        // Simple SGD step
        for (size_t i = 0; i < mlp.fc1.weights.size(); ++i) mlp.fc1.weights.data[i] -= lr * mlp.fc1.weights.grad[i];
        for (size_t i = 0; i < mlp.fc1.biases.size(); ++i)  mlp.fc1.biases.data[i]  -= lr * mlp.fc1.biases.grad[i];
        for (size_t i = 0; i < mlp.fc2.weights.size(); ++i) mlp.fc2.weights.data[i] -= lr * mlp.fc2.weights.grad[i];
        for (size_t i = 0; i < mlp.fc2.biases.size(); ++i)  mlp.fc2.biases.data[i]  -= lr * mlp.fc2.biases.grad[i];
    }

    std::cout << "  -> Initial Loss: " << initial_loss << " ---> Final Loss (200 steps): " << final_loss << std::endl;
    assert(final_loss < initial_loss * 0.05f); // Loss should drop by at least 95%!
    std::cout << "  -> Loss convergence verified! The 2-Layer MLP is learning! ✅\n";
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "        STARTING SPRINT 1: 2-LAYER MLP VERIFICATION          \n";
    std::cout << "============================================================\n\n";

    test_mlp_forward();
    test_finite_difference_gradcheck();
    test_training_convergence();

    std::cout << "\n============================================================\n";
    std::cout << " 🚀 SPRINT 1 COMPLETE! ALL MLP TESTS & GRADCHECKS PASSED! 🚀\n";
    std::cout << "============================================================\n";
    return 0;
}
