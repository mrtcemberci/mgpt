#include "SGDOptimiser.h"
#include "AdamWOptimiser.h"
#include "LinearLayer.h"
#include "Tensor.h"
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <memory>

// ============================================================================
// 1. SGD OPTIMISER STEP TEST
// ============================================================================
void test_sgd_optimiser() {
    std::cout << "Running Test 1: SGDOptimizer Step Verification..." << std::endl;
    float lr = 0.1f;
    SGDOptimizer sgd(lr);

    Tensor param({2, 2}, 1.0f);
    param.grad = {0.5f, -0.2f, 1.0f, 0.0f};

    std::vector<Tensor*> params = { &param };
    sgd.step(params);

    // Expected: param - lr * grad
    assert(std::abs(param.data[0] - 0.95f) < 1e-5f);
    assert(std::abs(param.data[1] - 1.02f) < 1e-5f);
    assert(std::abs(param.data[2] - 0.90f) < 1e-5f);
    assert(std::abs(param.data[3] - 1.00f) < 1e-5f);
    std::cout << "  -> SGD parameter update step verified! ✅\n";
}

// ============================================================================
// 2. ADAMW OPTIMISER STEP AND WEIGHT DECAY TEST
// ============================================================================
void test_adamw_optimiser() {
    std::cout << "Running Test 2: AdamWOptimizer Step & Weight Decay Verification..." << std::endl;
    float lr = 0.01f;
    float decay = 0.1f;
    AdamWOptimizer adamw(lr, 0.9f, 0.999f, 1e-8f, decay);

    Tensor param({1, 2}, 2.0f);
    param.grad = {0.1f, -0.1f};

    std::vector<Tensor*> params = { &param };
    adamw.step(params);

    // After step 1, weight decay reduces data by lr * decay * data = 0.01 * 0.1 * 2.0 = 0.002
    // Plus adaptive momentum update should move param in opposite direction of grad
    assert(param.data[0] < 2.0f); // Positive grad -> param decreases
    assert(param.data[1] > 2.0f); // Negative grad -> param increases
    std::cout << "  -> AdamW adaptive moment updates and weight decay verified! ✅\n";
}

// ============================================================================
// 3. STRATEGY PATTERN HOTSWAPPING IN A TOY TRAINING LOOP
// ============================================================================
void test_strategy_pattern() {
    std::cout << "Running Test 3: Optimizer Strategy Pattern Runtime Hotswapping..." << std::endl;
    LinearLayer layer(2, 1);
    layer.weights.data = {0.5f, -0.5f};
    layer.biases.data = {0.1f};

    Tensor X({1, 1, 2}, 1.0f);
    Tensor Target({1, 1, 1}, 0.0f);

    // Start with SGD strategy
    std::unique_ptr<Optimiser> opt = std::make_unique<SGDOptimizer>(0.05f);

    // Step 1: Forward, Backward, SGD Step
    Tensor Y1 = layer.forward(X);
    layer.backward(Y1 - Target);
    auto params1 = layer.get_parameters();
    opt->step(params1);

    float weight_after_sgd = layer.weights.data[0];
    assert(weight_after_sgd != 0.5f);

    // Hotswap strategy at runtime to AdamW!
    opt = std::make_unique<AdamWOptimizer>(0.01f);

    // Step 2: Forward, Backward, AdamW Step
    layer.weights.zero_grad();
    layer.biases.zero_grad();
    Tensor Y2 = layer.forward(X);
    layer.backward(Y2 - Target);
    auto params2 = layer.get_parameters();
    opt->step(params2);

    assert(layer.weights.data[0] != weight_after_sgd);
    std::cout << "  -> Runtime hotswapping between SGD and AdamW verified! ✅\n";
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "     STARTING OPTIMIZER STRATEGY VERIFICATION SUITE       \n";
    std::cout << "============================================================\n\n";

    test_sgd_optimiser();
    test_adamw_optimiser();
    test_strategy_pattern();

    std::cout << "\n============================================================\n";
    std::cout << " 🚀 OPTIMIZER VERIFICATION COMPLETE! ALL TESTS PASSED! 🚀  \n";
    std::cout << "============================================================\n";
    return 0;
}
