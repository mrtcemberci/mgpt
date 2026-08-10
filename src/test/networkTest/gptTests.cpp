#include "GPT.h"
#include "TransformerBlock.h"
#include "Tensor.h"
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <cstdio>

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

// ============================================================================
// 1. TRANSFORMER BLOCK FORWARD & BACKWARD SANITY CHECK
// ============================================================================
void test_transformer_block() {
    std::cout << "Running Test 1: TransformerBlock Forward & Backward Verification..." << std::endl;
    int channels = 4;
    TransformerBlock block(channels, 4 * channels, 1, 1, 1, false);

    Tensor input({2, 3, channels}, 0.1f);
    Tensor out = block.forward(input);
    assert(out.shape == input.shape);

    Tensor dout({2, 3, channels}, 0.05f);
    Tensor din = block.backward(dout);
    assert(din.shape == input.shape);

    std::cout << "  -> TransformerBlock forward/backward shapes and execution verified! ✅\n";
}

// ============================================================================
// 2. GPT FORWARD & LOSS COMPUTATION TEST
// ============================================================================
void test_gpt_forward_and_loss() {
    std::cout << "Running Test 2: GPT Forward Pass & Cross-Entropy Loss Verification..." << std::endl;
    GPTConfig config;
    config.vocab_size = 10;
    config.max_seq_len = 8;
    config.embed_dim = 6;
    config.num_layers = 2;

    GPT model(config);

    // Batch = 2, Time = 4
    Tensor input_ids({2, 4}, 0.0f);
    input_ids.data = {1.0f, 3.0f, 5.0f, 2.0f, 0.0f, 4.0f, 2.0f, 1.0f};

    Tensor target_ids({2, 4}, 0.0f);
    target_ids.data = {3.0f, 5.0f, 2.0f, 1.0f, 4.0f, 2.0f, 1.0f, 0.0f};

    Tensor logits = model.forward(input_ids);
    assert(logits.shape.size() == 3);
    assert(logits.shape[0] == 2 && logits.shape[1] == 4 && logits.shape[2] == 10);

    float loss = model.compute_loss(logits, target_ids);
    assert(loss > 0.0f && !std::isnan(loss));
    std::cout << "  -> GPT forward logits shape {2, 4, 10} and loss (" << loss << ") verified! ✅\n";
}

// ============================================================================
// 3. GPT BINARY WEIGHT SERIALIZATION (.bin) TEST
// ============================================================================
void test_gpt_binary_serialization() {
    std::cout << "Running Test 3: GPT Binary Weight Serialization (.bin) Verification..." << std::endl;
    GPTConfig config;
    config.vocab_size = 12;
    config.max_seq_len = 8;
    config.embed_dim = 4;
    config.num_layers = 1;
    config.num_heads = 2;

    GPT model1(config);

    // Modify a parameter weight in model1
    model1.get_parameters()[0]->data[0] = 42.42f;
    model1.get_parameters()[1]->data[2] = -99.99f;

    std::string test_bin_path = "test_model_weights.bin";
    model1.save_weights_bin(test_bin_path);

    // Create a fresh model2 with default weights
    GPT model2(config);
    assert(model2.get_parameters()[0]->data[0] != 42.42f);

    // Load weights from bin
    model2.load_weights_bin(test_bin_path);
    assert(std::abs(model2.get_parameters()[0]->data[0] - 42.42f) < 1e-5f);
    assert(std::abs(model2.get_parameters()[1]->data[2] - (-99.99f)) < 1e-5f);

    std::remove(test_bin_path.c_str());
    std::cout << "  -> GPT save_weights_bin and load_weights_bin verified! ✅\n";
}

// ============================================================================
// 4. GRADIENT CHECKPOINTING EXACT MATCH VERIFICATION
// ============================================================================
void test_gpt_gradient_checkpointing() {
    std::cout << "Running Test 4: GPT Gradient Checkpointing Equivalence Verification..." << std::endl;
    GPTConfig cfg_std;
    cfg_std.vocab_size = 12;
    cfg_std.max_seq_len = 8;
    cfg_std.embed_dim = 6;
    cfg_std.num_layers = 2;
    cfg_std.num_heads = 2;
    cfg_std.use_gradient_checkpointing = false;

    GPTConfig cfg_cp = cfg_std;
    cfg_cp.use_gradient_checkpointing = true;

    GPT model_std(cfg_std);
    GPT model_cp(cfg_cp);

    // Sync parameters
    auto p_std = model_std.get_parameters();
    auto p_cp = model_cp.get_parameters();
    for (size_t i = 0; i < p_std.size(); ++i) {
        p_cp[i]->data = p_std[i]->data;
    }

    Tensor input_ids({2, 4}, 0.0f);
    input_ids.data = {1.0f, 3.0f, 5.0f, 2.0f, 0.0f, 4.0f, 2.0f, 1.0f};

    Tensor target_ids({2, 4}, 0.0f);
    target_ids.data = {3.0f, 5.0f, 2.0f, 1.0f, 4.0f, 2.0f, 1.0f, 0.0f};

    Tensor logits_std = model_std.forward(input_ids);
    Tensor logits_cp = model_cp.forward(input_ids);

    float loss_std = model_std.compute_loss(logits_std, target_ids);
    float loss_cp = model_cp.compute_loss(logits_cp, target_ids);

    assert(std::abs(loss_std - loss_cp) < 1e-5f);

    // Verify parameter gradients match exactly
    for (size_t i = 0; i < p_std.size(); ++i) {
        assert(p_std[i]->grad.size() == p_cp[i]->grad.size());
        for (size_t j = 0; j < p_std[i]->grad.size(); ++j) {
            assert(std::abs(p_std[i]->grad[j] - p_cp[i]->grad[j]) < 1e-5f);
        }
    }
    std::cout << "  -> Gradient Checkpointing loss and gradients exactly match non-checkpointed execution! ✅\n";
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "        STARTING GPT ARCHITECTURE VERIFICATION SUITE       \n";
    std::cout << "============================================================\n\n";

    test_transformer_block();
    test_gpt_forward_and_loss();
    test_gpt_binary_serialization();
    test_gpt_gradient_checkpointing();

    std::cout << "\n============================================================\n";
    std::cout << "    🚀 GPT ARCHITECTURE VERIFICATION COMPLETE! PASSED! 🚀  \n";
    std::cout << "============================================================\n";
    return 0;
}
