#include "SingleHeadAttentionLayer.h"
#include "MultiHeadAttentionLayer.h"
#include "Tensor.h"
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>

bool check_grad_close(float analytic, float numerical, float tol = 5e-2f) {
    float diff = std::abs(analytic - numerical);
    float denom = std::abs(analytic) + std::abs(numerical) + 1e-3f;
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

// ============================================================================
// 4. SINGLE-HEAD ATTENTION _INTO PARITY & ZERO-ALLOCATION VERIFICATION
// ============================================================================
void test_sha_into_parity() {
    std::cout << "Running Test 4: SingleHeadAttentionLayer forward_into & backward_into Parity..." << std::endl;
    int B = 2, T = 3, C = 4;
    SingleHeadAttentionLayer attn(C);

    for (Tensor* p : attn.get_parameters()) {
        for (size_t i = 0; i < p->size(); ++i) {
            p->data[i] = ((float)(i % 5) - 2.0f) * 0.1f;
        }
    }

    Tensor X({B, T, C}, 0.0f);
    for (size_t i = 0; i < X.size(); ++i) X.data[i] = ((float)(i % 7) - 3.0f) * 0.2f;

    Tensor Target({B, T, C}, 0.0f);
    for (size_t i = 0; i < Target.size(); ++i) Target.data[i] = ((float)(i % 4) - 1.5f) * 0.3f;

    // Standard pass
    for (Tensor* p : attn.get_parameters()) p->zero_grad();
    Tensor Y_std = attn.forward(X);
    Tensor dY_std = compute_mse_grad(Y_std, Target);
    Tensor dX_std = attn.backward(dY_std);

    std::vector<Tensor> std_grads;
    for (Tensor* p : attn.get_parameters()) {
        std_grads.push_back(*p);
    }

    // _into pass
    for (Tensor* p : attn.get_parameters()) p->zero_grad();
    Tensor Y_into({B, T, C}, 0.0f);
    attn.forward_into(X, Y_into);

    for (size_t i = 0; i < Y_std.size(); ++i) {
        assert(std::abs(Y_std.data[i] - Y_into.data[i]) < 1e-5f);
    }
    std::cout << "  -> forward_into exactly matches standard forward! ✅\n";

    Tensor dY_into = compute_mse_grad(Y_into, Target);
    Tensor dX_into({B, T, C}, 0.0f);
    attn.backward_into(dY_into, dX_into);

    for (size_t i = 0; i < dX_std.size(); ++i) {
        assert(std::abs(dX_std.data[i] - dX_into.data[i]) < 1e-5f);
    }
    std::cout << "  -> backward_into input gradients (dX) match exactly! ✅\n";

    int idx = 0;
    for (Tensor* p : attn.get_parameters()) {
        for (size_t i = 0; i < p->size(); ++i) {
            assert(std::abs(p->grad[i] - std_grads[idx].grad[i]) < 1e-5f);
        }
        idx++;
    }
    std::cout << "  -> backward_into parameter gradients match exactly! ✅\n";
}

void test_flash_attention_cuda_parity() {
    std::cout << "Running Test 5: Flash Attention CUDA Parity Check..." << std::endl;
#ifndef USE_CUDA
    std::cout << "  -> CUDA not enabled, skipping test.\n";
    return;
#endif

    int B = 2, num_heads = 4, T = 128, head_dim = 64;
    Tensor Q({B * num_heads, T, head_dim}, 0.0f, Device::CPU);
    Tensor K({B * num_heads, T, head_dim}, 0.0f, Device::CPU);
    Tensor V({B * num_heads, T, head_dim}, 0.0f, Device::CPU);
    
    for (size_t i = 0; i < Q.size(); ++i) Q.data[i] = ((float)(i % 5) - 2.0f) * 0.1f;
    for (size_t i = 0; i < K.size(); ++i) K.data[i] = ((float)(i % 7) - 3.0f) * 0.2f;
    for (size_t i = 0; i < V.size(); ++i) V.data[i] = ((float)(i % 11) - 5.0f) * 0.1f;
    
    Tensor dO({B * num_heads, T, head_dim}, 0.0f, Device::CPU);
    for (size_t i = 0; i < dO.size(); ++i) dO.data[i] = ((float)(i % 3) - 1.0f) * 0.1f;

    // CPU standard forward
    Tensor K_T({B * num_heads, head_dim, T}, 0.0f, Device::CPU);
    K.transpose_into(1, 2, K_T);
    
    Tensor scores({B * num_heads, T, T}, 0.0f, Device::CPU);
    Q.matmul_into(K_T, scores);
    
    float scale = 1.0f / std::sqrt((float)head_dim);
    scores.mul_scalar_in_place(scale);
    scores.causal_mask();
    
    Tensor probs({B * num_heads, T, T}, 0.0f, Device::CPU);
    scores.softmax_into(-1, probs);
    
    Tensor O_cpu({B * num_heads, T, head_dim}, 0.0f, Device::CPU);
    probs.matmul_into(V, O_cpu);

    // CPU standard backward
    Tensor probs_T({B * num_heads, T, T}, 0.0f, Device::CPU);
    probs.transpose_into(1, 2, probs_T);
    
    Tensor dV_cpu({B * num_heads, T, head_dim}, 0.0f, Device::CPU);
    probs_T.matmul_into(dO, dV_cpu);
    
    Tensor V_T({B * num_heads, head_dim, T}, 0.0f, Device::CPU);
    V.transpose_into(1, 2, V_T);
    
    Tensor dP({B * num_heads, T, T}, 0.0f, Device::CPU);
    dO.matmul_into(V_T, dP);
    
    Tensor dS({B * num_heads, T, T}, 0.0f, Device::CPU);
    probs.softmax_backward_into(dP, dS);
    dS.mul_scalar_in_place(scale);
    
    Tensor dQ_cpu({B * num_heads, T, head_dim}, 0.0f, Device::CPU);
    dS.matmul_into(K, dQ_cpu);
    
    Tensor dS_T({B * num_heads, T, T}, 0.0f, Device::CPU);
    dS.transpose_into(1, 2, dS_T);
    
    Tensor dK_cpu({B * num_heads, T, head_dim}, 0.0f, Device::CPU);
    dS_T.matmul_into(Q, dK_cpu);

    // Flash Attention CUDA
    Q.shape = {B, num_heads, T, head_dim};
    K.shape = {B, num_heads, T, head_dim};
    V.shape = {B, num_heads, T, head_dim};
    dO.shape = {B, num_heads, T, head_dim};

    Tensor Q_gpu = Q; Q_gpu.to(Device::CUDA);
    Tensor K_gpu = K; K_gpu.to(Device::CUDA);
    Tensor V_gpu = V; V_gpu.to(Device::CUDA);
    Tensor dO_gpu = dO; dO_gpu.to(Device::CUDA);
    
    Tensor O_gpu({B, num_heads, T, head_dim}, 0.0f, Device::CUDA);
    Tensor L_gpu({B, num_heads, T}, 0.0f, Device::CUDA);
    
    Tensor::flash_attention_forward(Q_gpu, K_gpu, V_gpu, O_gpu, L_gpu, B, num_heads, T, head_dim);
    
    Tensor dQ_gpu({B, num_heads, T, head_dim}, 0.0f, Device::CUDA);
    Tensor dK_gpu({B, num_heads, T, head_dim}, 0.0f, Device::CUDA);
    Tensor dV_gpu({B, num_heads, T, head_dim}, 0.0f, Device::CUDA);
    
    Tensor::flash_attention_backward(Q_gpu, K_gpu, V_gpu, O_gpu, L_gpu, dO_gpu, dQ_gpu, dK_gpu, dV_gpu, B, num_heads, T, head_dim);

    O_gpu.to(Device::CPU);
    dQ_gpu.to(Device::CPU);
    dK_gpu.to(Device::CPU);
    dV_gpu.to(Device::CPU);

    for (size_t i = 0; i < O_cpu.size(); ++i) {
        assert(std::abs(O_cpu.data[i] - O_gpu.data[i]) < 1e-4f);
    }
    std::cout << "  -> Flash Forward perfectly matches Standard Forward! ✅\n";

    for (size_t i = 0; i < dV_cpu.size(); ++i) {
        assert(std::abs(dV_cpu.data[i] - dV_gpu.data[i]) < 1e-3f);
    }
    std::cout << "  -> Flash Backward dV perfectly matches Standard dV! ✅\n";

    for (size_t i = 0; i < dK_cpu.size(); ++i) {
        assert(std::abs(dK_cpu.data[i] - dK_gpu.data[i]) < 1e-3f);
    }
    std::cout << "  -> Flash Backward dK perfectly matches Standard dK! ✅\n";

    for (size_t i = 0; i < dQ_cpu.size(); ++i) {
        assert(std::abs(dQ_cpu.data[i] - dQ_gpu.data[i]) < 1e-3f);
    }
    std::cout << "  -> Flash Backward dQ perfectly matches Standard dQ! ✅\n";
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "      STARTING ATTENTION LAYER VERIFICATION SUITE         \n";
    std::cout << "============================================================\n\n";

    test_attention_forward();
    test_attention_gradcheck();
    test_mhsa();
    test_sha_into_parity();
    test_flash_attention_cuda_parity();

    std::cout << "\n============================================================\n";
    std::cout << " 🚀 ATTENTION VERIFICATION COMPLETE! ALL TESTS PASSED! 🚀  \n";
    std::cout << "============================================================\n";
    return 0;
}
