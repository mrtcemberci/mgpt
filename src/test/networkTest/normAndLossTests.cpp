#include "LayerNormLayer.h"
#include "CrossEntropyLossLayer.h"
#include "Tensor.h"
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>

// Helper for relative error in finite difference checking (using 3e-2f tolerance for float32 precision)
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
// 1. LAYER NORM FORWARD SANITY CHECK (Mean -> 0, Variance -> 1)
// ============================================================================
void test_layernorm_forward() {
    std::cout << "Running Test 1: LayerNorm Forward Pass Sanity Check..." << std::endl;
    int B = 2, T = 3, C = 4;
    LayerNormLayer norm(C, 1e-5f);

    Tensor X({B, T, C}, 0.0f);
    for (size_t i = 0; i < X.size(); ++i) X.data[i] = ((float)(i % 7) - 3.0f) * 2.0f;

    Tensor Y = norm.forward(X);
    assert(Y.shape == X.shape);

    int total_tokens = B * T;
    for (int i = 0; i < total_tokens; ++i) {
        const float* y_ptr = Y.data.data() + i * C;
        float sum = 0.0f;
        for (int j = 0; j < C; ++j) sum += y_ptr[j];
        float mean = sum / (float)C;
        assert(std::abs(mean) < 1e-5f); // Mean must be ~0

        float var_sum = 0.0f;
        for (int j = 0; j < C; ++j) var_sum += (y_ptr[j] - mean) * (y_ptr[j] - mean);
        float var = var_sum / (float)C;
        assert(std::abs(var - 1.0f) < 1e-3f); // Variance must be ~1
    }
    std::cout << "  -> LayerNorm zero mean and unit variance verified! ✅\n";
}

// ============================================================================
// 2. LAYER NORM SCALE AND SHIFT BEHAVIOR
// ============================================================================
void test_layernorm_scale_shift() {
    std::cout << "Running Test 2: LayerNorm Gamma/Beta Scale & Shift Behavior..." << std::endl;
    int C = 4;
    LayerNormLayer norm(C, 1e-5f);
    for (int j = 0; j < C; ++j) {
        norm.scale.data[j] = 3.0f;
        norm.shift.data[j] = 5.0f;
    }

    Tensor X({1, 1, C}, 0.0f);
    X.data = {10.0f, 20.0f, 30.0f, 40.0f};
    Tensor Y = norm.forward(X);

    float sum = 0.0f;
    for (int j = 0; j < C; ++j) sum += Y.data[j];
    float mean = sum / (float)C;
    assert(std::abs(mean - 5.0f) < 1e-4f); // Shifted mean must be 5.0

    float var_sum = 0.0f;
    for (int j = 0; j < C; ++j) var_sum += (Y.data[j] - mean) * (Y.data[j] - mean);
    float std_dev = std::sqrt(var_sum / (float)C);
    assert(std::abs(std_dev - 3.0f) < 1e-4f); // Scaled std dev must be 3.0
    std::cout << "  -> Gamma (scale) and Beta (shift) applied correctly! ✅\n";
}

// ============================================================================
// 3. LAYER NORM FINITE DIFFERENCE GRADCHECK (Scale, Shift, Input dX)
// ============================================================================
void test_layernorm_gradcheck() {
    std::cout << "Running Test 3: LayerNorm Finite Difference Numerical Gradcheck..." << std::endl;
    int B = 2, T = 2, C = 3;
    LayerNormLayer norm(C, 1e-5f);

    for (int j = 0; j < C; ++j) {
        norm.scale.data[j] = 0.5f * (float)(j + 1);
        norm.shift.data[j] = -0.2f * (float)(j + 1);
    }

    Tensor X({B, T, C}, 0.0f);
    for (size_t i = 0; i < X.size(); ++i) X.data[i] = ((float)(i % 5) - 2.0f) * 0.4f;

    Tensor Target({B, T, C}, 0.0f);
    for (size_t i = 0; i < Target.size(); ++i) Target.data[i] = ((float)(i % 3) - 1.0f) * 0.3f;

    // Analytic backward
    norm.scale.zero_grad();
    norm.shift.zero_grad();
    Tensor Y = norm.forward(X);
    Tensor dY = compute_mse_grad(Y, Target);
    Tensor dX_analytic = norm.backward(dY);

    float eps = 5e-3f;

    // Check scale (gamma) gradients
    std::cout << "  Checking scale (gamma) gradients...";
    for (int j = 0; j < C; ++j) {
        float orig = norm.scale.data[j];
        norm.scale.data[j] = orig + eps;
        float loss_plus = compute_mse_loss(norm.forward(X), Target);
        norm.scale.data[j] = orig - eps;
        float loss_minus = compute_mse_loss(norm.forward(X), Target);
        norm.scale.data[j] = orig;
        float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
        assert(check_grad_close(norm.scale.grad[j], num_grad));
    }
    std::cout << " Passed! ✅\n";

    // Check shift (beta) gradients
    std::cout << "  Checking shift (beta) gradients...";
    for (int j = 0; j < C; ++j) {
        float orig = norm.shift.data[j];
        norm.shift.data[j] = orig + eps;
        float loss_plus = compute_mse_loss(norm.forward(X), Target);
        norm.shift.data[j] = orig - eps;
        float loss_minus = compute_mse_loss(norm.forward(X), Target);
        norm.shift.data[j] = orig;
        float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
        assert(check_grad_close(norm.shift.grad[j], num_grad));
    }
    std::cout << " Passed! ✅\n";

    // Check input dX gradients
    std::cout << "  Checking input dX gradients...";
    for (size_t i = 0; i < X.size(); ++i) {
        float orig = X.data[i];
        X.data[i] = orig + eps;
        float loss_plus = compute_mse_loss(norm.forward(X), Target);
        X.data[i] = orig - eps;
        float loss_minus = compute_mse_loss(norm.forward(X), Target);
        X.data[i] = orig;
        float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
        assert(check_grad_close(dX_analytic.data[i], num_grad));
    }
    std::cout << " Passed! ✅\n";
}

// ============================================================================
// 4. CROSS ENTROPY FORWARD SANITY CHECK (Equal logits -> uniform probs)
// ============================================================================
void test_crossentropy_forward() {
    std::cout << "Running Test 4: CrossEntropyLoss Forward Pass Sanity Check..." << std::endl;
    CrossEntropyLossLayer loss_layer;
    Tensor logits({1, 1, 3}, 0.0f); // 3 classes with 0 logit each
    std::vector<int> targets = {0};

    float loss = loss_layer.forward_loss(logits, targets);
    float expected_loss = -std::log(1.0f / 3.0f);
    assert(std::abs(loss - expected_loss) < 1e-5f);
    std::cout << "  -> Uniform distribution loss (-log(1/3)) verified! ✅\n";
}

// ============================================================================
// 5. CROSS ENTROPY FINITE DIFFERENCE GRADCHECK
// ============================================================================
void test_crossentropy_gradcheck() {
    std::cout << "Running Test 5: CrossEntropyLoss Finite Difference Numerical Gradcheck..." << std::endl;
    CrossEntropyLossLayer loss_layer;
    int B = 2, T = 3, V = 4;
    Tensor logits({B, T, V}, 0.0f);
    for (size_t i = 0; i < logits.size(); ++i) logits.data[i] = ((float)(i % 7) - 3.0f) * 0.5f;

    std::vector<int> targets = {1, 0, 3, 2, 0, 1}; // B * T = 6 targets

    float orig_loss = loss_layer.forward_loss(logits, targets);
    (void)orig_loss;
    Tensor dL_analytic = loss_layer.backward(Tensor()); // Initiate backprop

    float eps = 5e-3f;
    std::cout << "  Checking logits dL gradients...";
    for (size_t i = 0; i < logits.size(); ++i) {
        float orig = logits.data[i];
        logits.data[i] = orig + eps;
        float loss_plus = loss_layer.forward_loss(logits, targets);
        logits.data[i] = orig - eps;
        float loss_minus = loss_layer.forward_loss(logits, targets);
        logits.data[i] = orig;
        float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
        assert(check_grad_close(dL_analytic.data[i], num_grad));
    }
    std::cout << " Passed! ✅\n";
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "    STARTING SPRINT 2: LAYERNORM & LOSS VERIFICATION       \n";
    std::cout << "============================================================\n\n";

    test_layernorm_forward();
    test_layernorm_scale_shift();
    test_layernorm_gradcheck();
    test_crossentropy_forward();
    test_crossentropy_gradcheck();

    std::cout << "\n============================================================\n";
    std::cout << " 🚀 SPRINT 2 COMPLETE! ALL NORM & LOSS TESTS PASSED! 🚀\n";
    std::cout << "============================================================\n";
    return 0;
}
