#include "SingleHeadAttentionLayer.h"
#include "MultiHeadAttentionLayer.h"
#include "Tensor.h"
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>

bool check_grad_close(float analytic, float numerical, float tol = 5e-2f) {
    float diff = std::abs(analytic - numerical);
    float denom = std::abs(analytic) + std::abs(numerical) + 1e-4f;
    float rel_error = diff / denom;
    if (rel_error >= tol) {
        std::cerr << "Grad check failed! Analytic: " << analytic 
                  << " vs Numerical: " << numerical << " (Rel Error: " << rel_error << ")\n";
        return false;
    }
    return true;
}

float compute_mse_loss(const Tensor& Y, const Tensor& T) {
    float sum = 0.0f;
    for (size_t i = 0; i < Y.size(); ++i) {
        float diff = Y.data[i] - T.data[i];
        sum += 0.5f * diff * diff;
    }
    return sum / (float)Y.size();
}

Tensor compute_mse_grad(const Tensor& Y, const Tensor& T) {
    Tensor dY(Y.shape, 0.0f);
    float scale = 1.0f / (float)Y.size();
    for (size_t i = 0; i < Y.size(); ++i) {
        dY.data[i] = (Y.data[i] - T.data[i]) * scale;
    }
    return dY;
}

// ============================================================================
// 1. ATTENTION FORWARD PASS SANITY CHECK
// ============================================================================
void test_attention_forward() {
    std::cout << "Running Test 1: SingleHeadAttentionLayer Forward Pass Sanity Check..." << std::endl;
    int channels = 4;
    SingleHeadAttentionLayer attn(channels);

    Tensor input({2, 3, channels}, 0.1f);
    Tensor Y = attn.forward(input);

    assert(Y.shape == input.shape);
    std::cout << "  -> Attention forward output shape verified! ✅\n";
}

// ============================================================================
// 2. ATTENTION FINITE DIFFERENCE GRADCHECK (W_Q, W_K, W_V, W_O, and dX)
// ============================================================================
void test_attention_gradcheck() {
    std::cout << "Running Test 2: SingleHeadAttentionLayer Finite Difference Gradcheck..." << std::endl;
    int B = 2, T = 3, C = 3;
    SingleHeadAttentionLayer attn(C);

    // Initialize weights with asymmetric small values
    for (Tensor* p : attn.get_parameters()) {
        for (size_t i = 0; i < p->size(); ++i) {
            p->data[i] = ((float)(i % 5) - 2.0f) * 0.15f;
        }
    }

    Tensor X({B, T, C}, 0.0f);
    for (size_t i = 0; i < X.size(); ++i) X.data[i] = ((float)(i % 7) - 3.0f) * 0.2f;

    Tensor Target({B, T, C}, 0.0f);
    for (size_t i = 0; i < Target.size(); ++i) Target.data[i] = ((float)(i % 4) - 1.5f) * 0.3f;

    // Zero out all parameters
    for (Tensor* p : attn.get_parameters()) p->zero_grad();

    Tensor Y = attn.forward(X);
    Tensor dY = compute_mse_grad(Y, Target);
    Tensor dX_analytic = attn.backward(dY);

    float eps = 5e-3f;

    // Helper lambda to check a specific projection layer
    auto check_layer_grads = [&](LinearLayer& layer, const std::string& name) {
        std::cout << "  Checking " << name << " weights...";
        for (size_t i = 0; i < layer.weights.size(); ++i) {
            float orig = layer.weights.data[i];
            layer.weights.data[i] = orig + eps;
            float loss_plus = compute_mse_loss(attn.forward(X), Target);
            layer.weights.data[i] = orig - eps;
            float loss_minus = compute_mse_loss(attn.forward(X), Target);
            layer.weights.data[i] = orig;

            float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
            assert(check_grad_close(layer.weights.grad[i], num_grad));
        }
        std::cout << " Passed! ✅\n";
    };

    check_layer_grads(attn.W_Q, "W_Q");
    check_layer_grads(attn.W_K, "W_K");
    check_layer_grads(attn.W_V, "W_V");
    check_layer_grads(attn.W_O, "W_O");

    std::cout << "  Checking input dX gradients...";
    for (size_t i = 0; i < X.size(); ++i) {
        float orig = X.data[i];
        X.data[i] = orig + eps;
        float loss_plus = compute_mse_loss(attn.forward(X), Target);
        X.data[i] = orig - eps;
        float loss_minus = compute_mse_loss(attn.forward(X), Target);
        X.data[i] = orig;

        float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
        assert(check_grad_close(dX_analytic.data[i], num_grad));
    }
    std::cout << " Passed! ✅\n";
}

// ============================================================================
// 3. MULTI-HEAD ATTENTION FORWARD & GRADCHECK
// ============================================================================
void test_mhsa() {
    std::cout << "Running Test 3: MultiHeadAttentionLayer Forward & Gradcheck..." << std::endl;
    int B = 2, T = 3, C = 4, num_heads = 2;
    MultiHeadAttentionLayer attn(C, num_heads);

    Tensor X({B, T, C}, 0.0f);
    for (size_t i = 0; i < X.size(); ++i) X.data[i] = (float)std::sin(i * 0.7f);

    Tensor Target({B, T, C}, 0.0f);
    for (size_t i = 0; i < Target.size(); ++i) Target.data[i] = (float)std::cos(i * 0.5f);

    for (Tensor* p : attn.get_parameters()) {
        for (size_t i = 0; i < p->size(); ++i) {
            p->data[i] = (float)std::sin(i * 0.3f + 0.1f) * 0.1f;
        }
    }

    Tensor Y = attn.forward(X);
    assert(Y.shape == X.shape);
    std::cout << "  -> MHSA forward output shape verified! ✅\n";

    Tensor dY = compute_mse_grad(Y, Target);
    Tensor dX_analytic = attn.backward(dY);

    float eps = 1e-3f;
    std::cout << "  Checking MHSA input dX gradients...";
    for (size_t i = 0; i < X.size(); ++i) {
        float orig = X.data[i];
        X.data[i] = orig + eps;
        float loss_plus = compute_mse_loss(attn.forward(X), Target);
        X.data[i] = orig - eps;
        float loss_minus = compute_mse_loss(attn.forward(X), Target);
        X.data[i] = orig;

        float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
        assert(check_grad_close(dX_analytic.data[i], num_grad));
    }
    std::cout << " Passed! ✅\n";
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "      STARTING ATTENTION LAYER VERIFICATION SUITE         \n";
    std::cout << "============================================================\n\n";

    test_attention_forward();
    test_attention_gradcheck();
    test_mhsa();

    std::cout << "\n============================================================\n";
    std::cout << " 🚀 ATTENTION VERIFICATION COMPLETE! ALL TESTS PASSED! 🚀  \n";
    std::cout << "============================================================\n";
    return 0;
}
