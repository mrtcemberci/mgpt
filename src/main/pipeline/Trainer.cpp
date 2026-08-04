#include "Trainer.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <memory>
#include "../utils/BytePairEncodingTokenizer.h"
#include "../neuralnetwork/optimisers/AdamWOptimiser.h"
#include "../utils/cuda_ops.h"

Trainer::Trainer(const CLIConfig& cfg) : config(cfg) {}

void Trainer::get_batch(const std::vector<int>& data, int batch_size, int max_seq_len, 
                        Tensor& x_batch, Tensor& y_batch, std::mt19937& rng, const int* d_data) {
    std::uniform_int_distribution<int> dist(0, (int)data.size() - max_seq_len - 1);
    
    if (x_batch.device == Device::CUDA && d_data) {
        std::vector<int> start_indices(batch_size);
        for (int b = 0; b < batch_size; ++b) {
            start_indices[b] = dist(rng);
        }
        static int* d_start_indices = nullptr;
        static int d_start_indices_size = 0;
        if (batch_size > d_start_indices_size) {
            if (d_start_indices) cuda_ops::free_int_memory(d_start_indices);
            cuda_ops::allocate_int_memory(&d_start_indices, batch_size);
            d_start_indices_size = batch_size;
        }
        cuda_ops::copy_int_host_to_device(d_start_indices, start_indices.data(), batch_size);
        cuda_ops::get_batch_gpu(d_data, (int)data.size(), batch_size, max_seq_len, d_start_indices, x_batch.get_data_ptr(), y_batch.get_data_ptr());
        return;
    }

    std::vector<float> x_tmp(batch_size * max_seq_len);
    std::vector<float> y_tmp(batch_size * max_seq_len);

    for (int b = 0; b < batch_size; ++b) {
        int idx = dist(rng);
        for (int t = 0; t < max_seq_len; ++t) {
            x_tmp[b * max_seq_len + t] = (float)data[idx + t];
            y_tmp[b * max_seq_len + t] = (float)data[idx + t + 1];
        }
    }
    x_batch.copy_from_host(x_tmp.data());
    y_batch.copy_from_host(y_tmp.data());
}

float Trainer::evaluate_loss(GPT& model, const std::vector<int>& val_data, 
                             int eval_steps, int batch_size, int max_seq_len, std::mt19937& rng, Device dev, const int* d_val_data) {
    Tensor x_batch({batch_size, max_seq_len}, 0.0f, dev);
    Tensor y_batch({batch_size, max_seq_len}, 0.0f, dev);
    Tensor logits({batch_size, max_seq_len, model.config.vocab_size}, 0.0f, dev);
    float total_loss = 0.0f;

    for (int step = 0; step < eval_steps; ++step) {
        get_batch(val_data, batch_size, max_seq_len, x_batch, y_batch, rng, d_val_data);
        model.forward_into(x_batch, logits);
        float step_loss = model.loss_layer.forward_loss(logits, y_batch);
        total_loss += step_loss;
        model.reset_activations();
    }
    return total_loss / (float)eval_steps;
}

int Trainer::run() {
    std::string data_path = config.data_path;
    std::string vocab_path = config.vocab_path;
    std::string weights_path = config.weights_path;

    // Locate dataset
    std::ifstream test_file(data_path);
    if (!test_file.is_open()) {
        if (data_path == "input.txt") {
            data_path = "../input.txt";
            test_file.open(data_path);
        }
        if (!test_file.is_open()) {
            std::cerr << "Error: Could not find dataset at " << data_path << "!\n";
            return -1;
        }
    }
    test_file.close();
    std::cout << "Using dataset at: " << data_path << "\n";

    BytePairEncodingTokenizer bpe_tokenizer(config.target_vocab_size);
    Tokenizer& tokenizer = bpe_tokenizer;

    std::string tok_bin_path = data_path + ".tok.bin";
    std::string vocab_bin_path = vocab_path.empty() ? (data_path + ".vocab.bin") : vocab_path;
    std::vector<int> full_data;

    std::ifstream tok_test(tok_bin_path, std::ios::binary);
    std::ifstream vocab_test(vocab_bin_path, std::ios::binary);
    bool vocab_loaded = vocab_test.is_open() && tokenizer.load_vocab(vocab_bin_path);
    if (vocab_test.is_open()) vocab_test.close();

    if (!vocab_path.empty() && !vocab_loaded) {
        std::cerr << "[WARNING]: Specified master vocabulary file '" << vocab_bin_path 
                  << "' could not be opened or loaded! Falling back to building vocabulary...\n";
    }

    if (tok_test.is_open() && vocab_loaded) {
        int loaded_vocab_size = (int)tokenizer.get_vocab_size();
        std::cout << "Found cached binary dataset files:\n"
                  << "      -> Loaded Vocabulary Size = " << loaded_vocab_size << " tokens (" << vocab_bin_path << ")\n"
                  << "      -> Loading pre-encoded token stream from " << tok_bin_path << "...\n";
        tok_test.seekg(0, std::ios::end);
        size_t file_bytes = tok_test.tellg();
        tok_test.seekg(0, std::ios::beg);
        size_t num_tokens = file_bytes / sizeof(int);
        full_data.resize(num_tokens);
        tok_test.read((char*)full_data.data(), file_bytes);
        tok_test.close();
        std::cout << "      Total Dataset Tokens: " << full_data.size() << " (Loaded instantly from binary cache!)\n";
    } else if (vocab_loaded) {
        if (tok_test.is_open()) tok_test.close();
        int loaded_vocab_size = (int)tokenizer.get_vocab_size();
        std::cout << "Found master vocabulary file (" << vocab_bin_path << "): " 
                  << loaded_vocab_size << " tokens.\n"
                  << "      -> Encoding dataset shard " << data_path << " using master vocabulary...\n";
        full_data = tokenizer.load_and_encode(data_path);
        std::cout << "      Total Dataset Shard Tokens: " << full_data.size() << "\n";
        std::cout << "      -> Caching raw encoded binary stream (" << (full_data.size() * sizeof(int)) / (1024 * 1024) 
                  << " MB) to " << tok_bin_path << "...\n";
        std::ofstream tok_out(tok_bin_path, std::ios::binary);
        if (tok_out.is_open()) {
            tok_out.write((const char*)full_data.data(), full_data.size() * sizeof(int));
            tok_out.close();
            std::cout << "      -> Successfully dumped raw encoded shard to " << tok_bin_path << "!\n";
        }
    } else {
        if (tok_test.is_open()) tok_test.close();

        std::cout << "Binary cache not found. Building Vocabulary from " << data_path << "...\n";
        tokenizer.build_vocab_from_file(data_path);
        int built_vocab_size = (int)tokenizer.get_vocab_size();
        std::cout << "      -> Vocabulary Built! Actual Vocab Size = " << built_vocab_size << " tokens.\n";

        std::cout << "      -> Encoding dataset text into token IDs...\n";
        full_data = tokenizer.load_and_encode(data_path);
        std::cout << "      Total Dataset Tokens: " << full_data.size() << "\n";

        std::cout << "      -> Caching vocabulary to " << vocab_bin_path << "...\n";
        tokenizer.save_vocab(vocab_bin_path);

        std::cout << "      -> Caching raw encoded binary stream (" << (full_data.size() * sizeof(int)) / (1024 * 1024) 
                  << " MB) to " << tok_bin_path << "...\n";
        std::ofstream tok_out(tok_bin_path, std::ios::binary);
        if (tok_out.is_open()) {
            tok_out.write((const char*)full_data.data(), full_data.size() * sizeof(int));
            tok_out.close();
            std::cout << "      -> Successfully dumped raw encoded data to " << tok_bin_path << "!\n";
        }
    }
    int vocab_size = (int)tokenizer.get_vocab_size();

    // Train / Validation Split
    std::vector<int> train_data, val_data;
    size_t split_idx = (size_t)(full_data.size() * 0.9);
    train_data.assign(full_data.begin(), full_data.begin() + split_idx);
    val_data.assign(full_data.begin() + split_idx, full_data.end());
    std::cout << "Train/Val Split (90/10):\n"
              << "      -> Training Tokens:   " << train_data.size() << "\n"
              << "      -> Validation Tokens: " << val_data.size() << "\n\n";

    // Configure & Instantiate GPT Model Architecture
    GPTConfig gpt_cfg;
    gpt_cfg.vocab_size = vocab_size;
    gpt_cfg.max_seq_len = config.max_seq_len;
    gpt_cfg.embed_dim = config.embed_dim;
    gpt_cfg.num_layers = config.num_layers;
    gpt_cfg.use_gradient_checkpointing = config.use_checkpointing;
    gpt_cfg.use_flash_attention = config.use_flash_attention;

    GPT model(gpt_cfg);

    size_t total_params = 0;
    for (Tensor* param : model.get_parameters()) {
        total_params += param->size();
    }
    std::cout << "Instantiated GPT Model Architecture:\n"
              << "      -> Vocab Size:    " << gpt_cfg.vocab_size << "\n"
              << "      -> Max Seq Len:   " << gpt_cfg.max_seq_len << "\n"
              << "      -> Embed Dim:     " << gpt_cfg.embed_dim << "\n"
              << "      -> Num Layers:    " << gpt_cfg.num_layers << "\n"
              << "      -> Checkpointing: " << (gpt_cfg.use_gradient_checkpointing ? "ENABLED (Block-Level Activation Recomputation)" : "DISABLED") << "\n"
              << "      -> Total Params:  " << total_params << " float32 parameters (~" 
              << (total_params * sizeof(float)) / 1024 << " KB)\n\n";

    std::mt19937 rng(67);

    Device target_dev = config.use_gpu ? Device::CUDA : Device::CPU;
    if (config.use_gpu) {
#ifndef USE_CUDA
        std::cerr << "Error: --gpu flag specified, but MGPT was compiled without CUDA support (-DUSE_CUDA=OFF)!\n";
        std::cerr << "Please rebuild with CMake option -DUSE_CUDA=ON.\n";
        return -1;
#endif
        std::cout << "Migrating GPT Model and Engine to CUDA GPU...\n";
        model.to(target_dev);

        std::cout << "Running Mathematical VRAM Profiler...\n";
        size_t B = config.batch_size;
        size_t T = config.max_seq_len;
        size_t C = config.embed_dim;
        size_t H = gpt_cfg.num_heads;
        size_t Vocab = config.target_vocab_size;
        size_t top_k = gpt_cfg.top_k;
        size_t num_experts = gpt_cfg.num_experts;
        
        // TransformerBlock Peak (Forward standing + Backward standing + MHA + Linear)
        // MHA Fwd standing: 4 * B*T*C
        // MoE Fwd standing: 14 * top_k * B*T*C
        size_t block_fwd_standing = (4 + 14 * top_k) * B * T * C;
        if (!config.use_checkpointing) {
            block_fwd_standing *= config.num_layers;
        }

        // MHA Bwd standing: 4 * B*T*C
        // MoE Bwd standing: 20 * top_k * B*T*C
        size_t block_bwd_standing = (4 + 20 * top_k) * B * T * C;
        size_t mha_bwd = 9 * B * T * C + 4 * B * H * T * T;
        size_t w_qkv_bwd = 3 * C + 3 * B * T * C + 6 * C * C;
        
        // MoE Router Backward Scratchpad: d_top_probs + d_router_probs + lb_grads_tensor + d_router_logits + router_din
        size_t moe_router_bwd = B * T * top_k + B * T * num_experts * 2 + num_experts + B * T * C;
        
        size_t block_peak = block_fwd_standing + block_bwd_standing + mha_bwd + w_qkv_bwd + moe_router_bwd;

        // LM Head Backward Peak
        size_t lm_head_peak = B * T * C + 2 * C * Vocab + Vocab;

        // Global peak is the max of the deep network branches
        size_t peak_floats = std::max(block_peak, lm_head_peak);
        size_t peak_mb = (peak_floats * sizeof(float)) / (1024 * 1024);

        std::cout << "      -> Dynamic Scratchpad Peak: " << peak_floats << " floats (~" << peak_mb << " MB)\n";
        
        if (config.mode_virtual) {
            std::cout << "      -> Peak VRAM calculated successfully. Exiting virtual mode.\n";
            return 0;
        }

        // Allocate real scratchpad with a 5% safety buffer
        size_t safe_capacity = (size_t)(peak_floats * 1.05f);
        model.init_scratchpad(safe_capacity);
    } else if (config.mode_virtual) {
        std::cout << "Virtual mode (--virtual) is only supported with --gpu flag.\n";
        return 0;
    }

    int start_step = config.start_step;
    if (config.mode_resume) {
        std::cout << "Resuming Training Mode: Loading existing checkpoint from " << weights_path << "...\n";
        int loaded_steps = model.load_weights_bin(weights_path);
        if (start_step == -1) start_step = loaded_steps;
        if (config.use_gpu) model.to(target_dev);
    }
    if (start_step == -1) start_step = 0;

    int max_steps = config.max_steps;
    int eval_interval = std::max(1, max_steps / 10);
    int print_interval = std::max(1, max_steps / 10);
    int eval_steps = 10;

    std::unique_ptr<Optimiser> optimizer = std::make_unique<AdamWOptimizer>(config.learning_rate);

    int* d_train_data = nullptr;
    int* d_val_data = nullptr;
    if (target_dev == Device::CUDA) {
        std::cout << "Uploading Train and Validation Datasets to GPU Memory...\n";
        cuda_ops::allocate_int_memory(&d_train_data, train_data.size());
        cuda_ops::copy_int_host_to_device(d_train_data, train_data.data(), train_data.size());
        cuda_ops::allocate_int_memory(&d_val_data, val_data.size());
        cuda_ops::copy_int_host_to_device(d_val_data, val_data.data(), val_data.size());
    }

    int batch_size = config.batch_size;
    int grad_accum_steps = config.grad_accum_steps;
    Tensor x_batch({batch_size, gpt_cfg.max_seq_len}, 0.0f, target_dev);
    Tensor y_batch({batch_size, gpt_cfg.max_seq_len}, 0.0f, target_dev);
    Tensor logits({batch_size, gpt_cfg.max_seq_len, gpt_cfg.vocab_size}, 0.0f, target_dev);
    std::vector<Tensor*> params = model.get_parameters();
    if (config.mode_resume) {
        std::string opt_path = weights_path + ".opt";
        if (auto* adamw = dynamic_cast<AdamWOptimizer*>(optimizer.get())) {
            adamw->load_state(opt_path, params);
            adamw->set_step(start_step);
        }
    }

    int global_total_steps = (config.total_steps > 0) ? config.total_steps : (start_step + max_steps);

    std::cout << "Starting Training Loop (AdamW, LR=" << config.learning_rate 
              << ", Micro-Batch=" << batch_size
              << ", Accum Steps=" << grad_accum_steps
              << " [Effective Batch: " << (batch_size * grad_accum_steps) << "]"
              << ", Shard Steps=" << max_steps
              << ", Global Horizon=" << start_step << " -> " << (start_step + max_steps) << "/" << global_total_steps << ")...\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Step       Progress & Timings                      Train Loss   Val Loss\n";
    std::cout << "------------------------------------------------------------\n";

    auto calculate_lr = [](int step, int max_steps, float max_lr, float min_lr) -> float {
        float PI = 3.14159265358979323846f;
        float warmup_steps = max_steps * 0.05f;
        if (step < warmup_steps) {
            return max_lr * (step / warmup_steps);
        }
        float decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps);
        decay_ratio = std::min(1.0f, std::max(0.0f, decay_ratio));
        return min_lr + 0.5f * (max_lr - min_lr) * (1.0f + std::cos(PI * decay_ratio));
    };

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = 1; step <= max_steps; ++step) {
        int current_global_step = start_step + step;
        float current_lr = calculate_lr(current_global_step, global_total_steps, config.learning_rate, config.learning_rate * 0.1f);

        optimizer->set_lr(current_lr);

        auto step_start = std::chrono::high_resolution_clock::now();

        for (Tensor* param : params) {
            param->zero_grad();
        }

        float accum_train_loss = 0.0f;
        double total_fwd_ms = 0.0;
        double total_bwd_ms = 0.0;

        for (int micro = 0; micro < grad_accum_steps; ++micro) {
            get_batch(train_data, batch_size, gpt_cfg.max_seq_len, x_batch, y_batch, rng, d_train_data);

            auto t0 = std::chrono::high_resolution_clock::now();
            model.forward_into(x_batch, logits);
            auto t1 = std::chrono::high_resolution_clock::now();

            float micro_loss = model.compute_loss(logits, y_batch);
            auto t2 = std::chrono::high_resolution_clock::now();

            accum_train_loss += micro_loss;
            total_fwd_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_bwd_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }

        if (grad_accum_steps > 1) {
            float inv_accum = 1.0f / (float)grad_accum_steps;
            for (Tensor* param : params) {
                if (!param) continue;
                if (param->device == Device::CUDA) {
                    cuda_ops::scale_inplace(param->get_grad_ptr(), inv_accum, (int)param->size());
                } else {
                    float* gptr = param->get_grad_ptr();
                    for (size_t i = 0; i < param->size(); ++i) {
                        gptr[i] *= inv_accum;
                    }
                }
            }
        }

        float train_loss = accum_train_loss / (float)grad_accum_steps;

        auto opt_start = std::chrono::high_resolution_clock::now();
        float max_grad_norm = 1.0f;
        double total_sq_norm = 0.0;
        
        if (target_dev == Device::CUDA) {
            cuda_ops::reset_sq_norm();
        }
        
        for (Tensor* param : params) {
            if (!param) continue;
            if (param->device == Device::CUDA) {
                cuda_ops::accumulate_sq_norm(param->get_grad_ptr(), (int)param->size());
            } else {
                const float* gptr = param->get_grad_ptr();
                for (size_t i = 0; i < param->size(); ++i) {
                    total_sq_norm += (double)gptr[i] * gptr[i];
                }
            }
        }
        
        if (target_dev == Device::CUDA) {
            total_sq_norm = (double)cuda_ops::get_sq_norm();
        }
        float total_norm = (float)std::sqrt(total_sq_norm);
        if (std::isnan(total_norm) || std::isinf(total_norm)) {
            for (Tensor* param : params) {
                if (param) param->zero_grad();
            }
        } else {
            if (total_norm > max_grad_norm) {
                float scale = max_grad_norm / (total_norm + 1e-6f);
                for (Tensor* param : params) {
                    if (!param) continue;
                    if (param->device == Device::CUDA) {
                        cuda_ops::scale_inplace(param->get_grad_ptr(), scale, (int)param->size());
                    } else {
                        float* gptr = param->get_grad_ptr();
                        for (size_t i = 0; i < param->size(); ++i) {
                            gptr[i] *= scale;
                        }
                    }
                }
            }
            optimizer->step(params);
        }
        auto opt_end = std::chrono::high_resolution_clock::now();
        double opt_ms = std::chrono::duration<double, std::milli>(opt_end - opt_start).count();

        auto step_end = std::chrono::high_resolution_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();

        if (step % print_interval == 0 || step == 1 || step == max_steps) {
            float val_loss = 0.0f;
            if (step % eval_interval == 0 || step == max_steps) {
                val_loss = evaluate_loss(model, val_data, eval_steps, batch_size, gpt_cfg.max_seq_len, rng, target_dev, d_val_data);
            }

            int percent = (int)((current_global_step * 100.0) / global_total_steps);
            std::cout << "[" << std::setw(4) << current_global_step << "/" << global_total_steps 
                      << " (Shard " << step << "/" << max_steps << " | " << std::setw(3) << percent << "%)] "
                      << "LR:" << std::scientific << std::setprecision(2) << current_lr << " "
                      << "Fwd:" << (int)total_fwd_ms << "ms Loss:" << (int)total_bwd_ms << "ms Opt:" << (int)opt_ms << "ms "
                      << "| Step:" << (int)step_ms << "ms | Loss: " 
                      << std::fixed << std::setprecision(4) << train_loss;
                      
            if (!model.blocks.empty() && gpt_cfg.num_experts > 0) {
                std::cout << " | Experts: [";
                auto counts = model.blocks[0]->mlp.get_expert_counts();
                for (int e = 0; e < gpt_cfg.num_experts; e++) {
                    std::cout << "E" << e << ":" << (int)counts[e] << (e == gpt_cfg.num_experts - 1 ? "" : " ");
                }
                std::cout << "]";
            }

            if (val_loss > 0.0f) {
                std::cout << " | Val: " << val_loss;
            }
            std::cout << "\n" << std::flush;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Training Complete! Total Duration: " << std::fixed << std::setprecision(2) << total_sec << " seconds.\n\n";

    if (d_train_data) cuda_ops::free_int_memory(d_train_data);
    if (d_val_data) cuda_ops::free_int_memory(d_val_data);

    std::cout << "Exporting Trained Model to " << weights_path << "...\n";
    model.save_weights_bin(weights_path, start_step + max_steps);
    if (auto* adamw = dynamic_cast<AdamWOptimizer*>(optimizer.get())) {
        adamw->save_state(weights_path + ".opt", params);
    }

    return 0;
}
