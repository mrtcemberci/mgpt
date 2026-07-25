#include "Generator.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include "../utils/BytePairEncodingTokenizer.h"

Generator::Generator(const CLIConfig& cfg) : config(cfg) {}

std::string Generator::generate_text(GPT& model, Tokenizer& tokenizer, const std::string& prompt, 
                                     int max_new_tokens, float temperature, int top_k, std::mt19937& rng) {
    std::vector<int> tokens = tokenizer.encode(prompt);
    
    for (int step = 0; step < max_new_tokens; ++step) {
        // Crop context to max_seq_len if needed
        int start_idx = 0;
        if ((int)tokens.size() > model.config.max_seq_len) {
            start_idx = (int)tokens.size() - model.config.max_seq_len;
        }
        int seq_len = (int)tokens.size() - start_idx;

        Tensor input_ids({1, seq_len}, 0.0f, model.tok_emb.lookup_table.device);
        std::vector<float> tmp_ids(seq_len);
        for (int t = 0; t < seq_len; ++t) {
            tmp_ids[t] = (float)tokens[start_idx + t];
        }
        input_ids.copy_from_host(tmp_ids.data());

        // Forward pass
        Tensor logits = model.forward(input_ids);
        model.reset_activations();

        std::vector<float> logits_host(logits.size());
        logits.copy_to_host(logits_host.data());

        // Get softmax probabilities for the last time step
        int last_t_offset = (seq_len - 1) * model.config.vocab_size;
        
        // Apply Temperature Scaling
        float temp = (temperature < 1e-4f) ? 1e-4f : temperature;
        for (int v = 0; v < model.config.vocab_size; ++v) {
            logits_host[last_t_offset + v] /= temp;
        }

        // Apply Top-K Filtering (Clamp low-probability tail tokens to -1e15)
        if (top_k > 0 && top_k < model.config.vocab_size) {
            std::vector<float> sorted_logits(model.config.vocab_size);
            for (int v = 0; v < model.config.vocab_size; ++v) {
                sorted_logits[v] = logits_host[last_t_offset + v];
            }
            std::sort(sorted_logits.rbegin(), sorted_logits.rend());
            float cutoff = sorted_logits[top_k - 1];
            for (int v = 0; v < model.config.vocab_size; ++v) {
                if (logits_host[last_t_offset + v] < cutoff) {
                    logits_host[last_t_offset + v] = -1e15f;
                }
            }
        }

        // Find max logit for stable softmax
        float max_logit = -1e15f;
        for (int v = 0; v < model.config.vocab_size; ++v) {
            if (logits_host[last_t_offset + v] > max_logit) {
                max_logit = logits_host[last_t_offset + v];
            }
        }

        // Compute softmax probabilities
        std::vector<float> probs(model.config.vocab_size);
        float sum_exp = 0.0f;
        for (int v = 0; v < model.config.vocab_size; ++v) {
            probs[v] = std::exp(logits_host[last_t_offset + v] - max_logit);
            sum_exp += probs[v];
        }
        for (int v = 0; v < model.config.vocab_size; ++v) {
            probs[v] /= sum_exp;
        }

        // Sample next token ID from probability distribution
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        int next_token = dist(rng);
        tokens.push_back(next_token);
    }

    return tokenizer.decode(tokens);
}

int Generator::run() {
    std::string data_path = config.data_path;
    std::string vocab_path = config.vocab_path;
    std::string weights_path = config.weights_path;

    // Locate dataset or vocabulary
    BytePairEncodingTokenizer bpe_tokenizer(config.target_vocab_size);
    Tokenizer& tokenizer = bpe_tokenizer;

    std::string vocab_bin_path = vocab_path.empty() ? (data_path + ".vocab.bin") : vocab_path;
    std::ifstream vocab_test(vocab_bin_path, std::ios::binary);
    bool vocab_loaded = vocab_test.is_open() && tokenizer.load_vocab(vocab_bin_path);
    if (vocab_test.is_open()) vocab_test.close();

    if (!vocab_loaded) {
        std::cerr << "[WARNING]: Could not load vocabulary from " << vocab_bin_path 
                  << "! Attempting to build or load from data path " << data_path << "...\n";
        std::ifstream tok_test(data_path + ".tok.bin", std::ios::binary);
        if (!tok_test.is_open()) {
            tokenizer.build_vocab_from_file(data_path);
        } else {
            tok_test.close();
        }
    }
    int vocab_size = (int)tokenizer.get_vocab_size();

    // Configure & Instantiate GPT Model Architecture
    GPTConfig gpt_cfg;
    gpt_cfg.vocab_size = vocab_size;
    gpt_cfg.max_seq_len = config.max_seq_len;
    gpt_cfg.embed_dim = config.embed_dim;
    gpt_cfg.num_layers = config.num_layers;
    gpt_cfg.use_gradient_checkpointing = config.use_checkpointing;
    gpt_cfg.use_flash_attention = config.use_flash_attention;

    GPT model(gpt_cfg);

    std::cout << "Instantiated GPT Model Architecture (Inference Mode):\n"
              << "      -> Vocab Size:    " << gpt_cfg.vocab_size << "\n"
              << "      -> Max Seq Len:   " << gpt_cfg.max_seq_len << "\n"
              << "      -> Embed Dim:     " << gpt_cfg.embed_dim << "\n"
              << "      -> Num Layers:    " << gpt_cfg.num_layers << "\n\n";

    Device target_dev = config.use_gpu ? Device::CUDA : Device::CPU;
    if (config.use_gpu) {
#ifndef USE_CUDA
        std::cerr << "Error: --gpu flag specified, but MGPT was compiled without CUDA support (-DUSE_CUDA=OFF)!\n";
        return -1;
#endif
        std::cout << "Migrating GPT Model to CUDA GPU...\n";
        model.to(target_dev);
        model.init_scratchpad(256 * 1024 * 1024);
    }

    std::cout << "Loading Trained Model Weights from " << weights_path << "...\n";
    model.load_weights_bin(weights_path);
    if (config.use_gpu) model.to(target_dev);

    std::mt19937 rng(67);
    std::cout << "\n--- Text Generation Sample (Prompt: \"" << config.prompt << "\" | Temp: " << config.temperature << " | Top-K: " << config.top_k << ") ---\n";
    std::string generated = generate_text(model, tokenizer, config.prompt, config.max_tokens, config.temperature, config.top_k, rng);
    std::cout << generated << "\n";
    std::cout << "------------------------------------------------------------\n\n";

    return 0;
}
