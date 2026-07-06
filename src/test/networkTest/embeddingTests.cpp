#include "EmbeddingLayer.h"
#include "Tensor.h"
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>

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
// 1. EMBEDDING FORWARD PASS SANITY CHECK
// ============================================================================
void test_embedding_forward() {
    std::cout << "Running Test 1: EmbeddingLayer Forward Pass Sanity Check..." << std::endl;
    int vocab_size = 10;
    int embed_dim = 4;
    EmbeddingLayer emb(vocab_size, embed_dim);

    // Set known values in lookup table
    for (int k = 0; k < vocab_size; ++k) {
        for (int j = 0; j < embed_dim; ++j) {
            emb.lookup_table.data[k * embed_dim + j] = (float)(k * 10 + j);
        }
    }

    Tensor input({2, 3}, 0.0f); // Batch = 2, Time = 3
    input.data = {1.0f, 5.0f, 2.0f, 5.0f, 0.0f, 9.0f};

    Tensor Y = emb.forward(input);
    assert(Y.shape.size() == 3);
    assert(Y.shape[0] == 2 && Y.shape[1] == 3 && Y.shape[2] == 4);

    // Check lookup values (e.g. token 5 at index 1 and 3 should copy row 5)
    for (int j = 0; j < embed_dim; ++j) {
        assert(std::abs(Y.data[1 * embed_dim + j] - (50.0f + j)) < 1e-5f);
        assert(std::abs(Y.data[3 * embed_dim + j] - (50.0f + j)) < 1e-5f);
    }
    std::cout << "  -> Embedding lookup table row indexing and shape verified! ✅\n";
}

// ============================================================================
// 2. EMBEDDING FINITE DIFFERENCE GRADCHECK (Accumulation over repeated tokens)
// ============================================================================
void test_embedding_gradcheck() {
    std::cout << "Running Test 2: EmbeddingLayer Finite Difference Gradcheck..." << std::endl;
    int vocab_size = 6;
    int embed_dim = 3;
    EmbeddingLayer emb(vocab_size, embed_dim);

    for (size_t i = 0; i < emb.lookup_table.size(); ++i) {
        emb.lookup_table.data[i] = ((float)(i % 7) - 3.0f) * 0.4f;
    }

    Tensor input({1, 4}, 0.0f);
    input.data = {2.0f, 1.0f, 2.0f, 4.0f}; // Token 2 repeated twice!

    Tensor Target({1, 4, embed_dim}, 0.0f);
    for (size_t i = 0; i < Target.size(); ++i) Target.data[i] = ((float)(i % 5) - 2.0f) * 0.3f;

    emb.lookup_table.zero_grad();
    Tensor Y = emb.forward(input);
    Tensor dY = compute_mse_grad(Y, Target);
    emb.backward(dY);

    float eps = 5e-3f;
    std::cout << "  Checking lookup table dW gradients...";
    for (size_t i = 0; i < emb.lookup_table.size(); ++i) {
        float orig = emb.lookup_table.data[i];
        emb.lookup_table.data[i] = orig + eps;
        float loss_plus = compute_mse_loss(emb.forward(input), Target);
        emb.lookup_table.data[i] = orig - eps;
        float loss_minus = compute_mse_loss(emb.forward(input), Target);
        emb.lookup_table.data[i] = orig;

        float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
        assert(check_grad_close(emb.lookup_table.grad[i], num_grad));
    }
    std::cout << " Passed! ✅\n";
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "      STARTING EMBEDDING LAYER VERIFICATION SUITE         \n";
    std::cout << "============================================================\n\n";

    test_embedding_forward();
    test_embedding_gradcheck();

    std::cout << "\n============================================================\n";
    std::cout << " 🚀 EMBEDDING VERIFICATION COMPLETE! ALL TESTS PASSED! 🚀  \n";
    std::cout << "============================================================\n";
    return 0;
}
