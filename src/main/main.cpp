#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include "utils/Tokenizer.h"
#include "utils/CharacterTokenizer.h"
#include "utils/BytePairEncodingTokenizer.h"
#include "neuralnetwork/GPT.h"
#include "neuralnetwork/optimisers/AdamWOptimiser.h"
#include "utils/cuda_ops.h"

void get_batch(const std::vector<int>& data, int batch_size, int max_seq_len, 
               Tensor& x_batch, Tensor& y_batch, std::mt19937& rng, const int* d_data = nullptr) {
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

// Evaluate average loss on validation dataset without backprop
float evaluate_loss(GPT& model, const std::vector<int>& val_data, 
                    int eval_steps, int batch_size, int max_seq_len, std::mt19937& rng, Device dev, const int* d_val_data = nullptr) {
    Tensor x_batch({batch_size, max_seq_len}, 0.0f, dev);
    Tensor y_batch({batch_size, max_seq_len}, 0.0f, dev);
    Tensor logits({batch_size, max_seq_len, model.config.vocab_size}, 0.0f, dev);
    float total_loss = 0.0f;

    for (int step = 0; step < eval_steps; ++step) {
        get_batch(val_data, batch_size, max_seq_len, x_batch, y_batch, rng, d_val_data);
        model.forward_into(x_batch, logits);
        float step_loss = model.loss_layer.forward_loss(logits, y_batch);
        total_loss += step_loss;
    }
    return total_loss / (float)eval_steps;
}

//  Autoregressive text generation sample from trained model
std::string generate_text(GPT& model, Tokenizer& tokenizer, const std::string& prompt, 
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


int main(int argc, char** argv) {
    try {
        std::cout << "============================================================\n";
    std::cout << "      🚀 MGPT: BPE TOKENIZED GPT TRAINING ENGINE 🚀       \n";
    std::cout << "============================================================\n\n";

    bool mode_train = true;  // default to train if neither -t nor -i specified
    bool mode_infer_only = false;
    bool use_gpu = false;
    std::string weights_path = "shakespeare_gpt.bin";
    std::string data_path = "input.txt";
    std::string prompt = "To be or not to be";
    int max_tokens = 500;
    int max_steps = 1000;
    int num_layers = 4;
    int embed_dim = 128;
    int batch_size = 16;
    int max_seq_len = 64;
    int target_vocab_size = 512;
    float temperature = 0.8f;
    int top_k = 15;

    int grad_accum_steps = 1;

    auto strip_quotes = [](std::string& s) {
        if (s.length() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
            s = s.substr(1, s.length() - 2);
        }
    };
    bool use_checkpointing = true; // Enabled by default to maximize VRAM efficiency

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: mgpt [options]\n"
                      << "Options:\n"
                      << "  -t, --train               Run in training mode (default)\n"
                      << "  -i, --infer               Run in inference/generation mode (load weights without training)\n"
                      << "  -g, --gpu                 Run training and inference using CUDA GPU engine\n"
                      << "  -k, --checkpointing       Enable gradient/activation checkpointing (default: enabled)\n"
                      << "      --no-checkpointing    Disable gradient/activation checkpointing\n"
                      << "  -f, --file <path>         Path to model weights file (.bin) (default: shakespeare_gpt.bin)\n"
                      << "  -d, --data <path>         Path to training dataset (default: input.txt)\n"
                      << "  -v, --vocab <int>         Target BPE vocabulary size (default: 512)\n"
                      << "  -T, --temp <float>        Sampling temperature for text generation (default: 0.8)\n"
                      << "  -K, --topk <int>          Top-K nucleus filtering for generation (default: 15)\n"
                      << "  -p, --prompt <str>        Text generation prompt (default: \"To be or not to be\")\n"
                      << "  -n, --tokens <int>        Number of characters to generate (default: 500)\n"
                      << "  -s, --steps <int>         Number of training steps (default: 1000)\n"
                      << "  -l, --layers <int>        Number of Transformer blocks (default: 4)\n"
                      << "  -c, --channels <int>      Embedding channel dimension (default: 128)\n"
                      << "  -b, --batch <int>         Training batch size (default: 16)\n"
                      << "  -w, --window <int>        Context window / sequence length (default: 64)\n"
                      << "  -a, --accumulate <int>    Gradient accumulation steps (default: 1)\n"
                      << "  -h, --help                Show this help message and exit\n\n"
                      << "Examples:\n"
                      << "  mgpt -t -g -v=256 -f=\"my_model.bin\" -s=2000 -l=6 -b=64 -w=128\n"
                      << "  mgpt -i -g -T=0.7 -f=\"my_model.bin\" -p=\"O Romeo, Romeo!\" -n=1000\n";
            return 0;
        }
        if (arg == "-t" || arg == "--train") { mode_train = true; mode_infer_only = false; continue; }
        if (arg == "-i" || arg == "--infer") { mode_infer_only = true; mode_train = false; continue; }
        if (arg == "-g" || arg == "--gpu") { use_gpu = true; continue; }
        if (arg == "-k" || arg == "--checkpointing") { use_checkpointing = true; continue; }
        if (arg == "--no-checkpointing") { use_checkpointing = false; continue; }
        
        if (arg == "-f" || arg == "--file") { if (i + 1 < argc) weights_path = argv[++i]; continue; }
        if (arg.find("-f=") == 0) { weights_path = arg.substr(3); continue; }
        if (arg.find("--file=") == 0) { weights_path = arg.substr(7); continue; }

        if (arg == "-d" || arg == "--data") { if (i + 1 < argc) data_path = argv[++i]; continue; }
        if (arg.find("-d=") == 0) { data_path = arg.substr(3); continue; }
        if (arg.find("--data=") == 0) { data_path = arg.substr(7); continue; }

        if (arg == "-p" || arg == "--prompt") { if (i + 1 < argc) prompt = argv[++i]; continue; }
        if (arg.find("-p=") == 0) { prompt = arg.substr(3); continue; }
        if (arg.find("--prompt=") == 0) { prompt = arg.substr(9); continue; }

        if (arg == "-n" || arg == "--tokens") { if (i + 1 < argc) max_tokens = std::stoi(argv[++i]); continue; }
        if (arg.find("-n=") == 0) { max_tokens = std::stoi(arg.substr(3)); continue; }
        if (arg.find("--tokens=") == 0) { max_tokens = std::stoi(arg.substr(9)); continue; }

        if (arg == "-s" || arg == "--steps") { if (i + 1 < argc) max_steps = std::stoi(argv[++i]); continue; }
        if (arg.find("-s=") == 0) { max_steps = std::stoi(arg.substr(3)); continue; }
        if (arg.find("--steps=") == 0) { max_steps = std::stoi(arg.substr(8)); continue; }

        if (arg == "-l" || arg == "--layers") { if (i + 1 < argc) num_layers = std::stoi(argv[++i]); continue; }
        if (arg.find("-l=") == 0) { num_layers = std::stoi(arg.substr(3)); continue; }
        if (arg.find("--layers=") == 0) { num_layers = std::stoi(arg.substr(9)); continue; }

        if (arg == "-c" || arg == "--channels") { if (i + 1 < argc) embed_dim = std::stoi(argv[++i]); continue; }
        if (arg.find("-c=") == 0) { embed_dim = std::stoi(arg.substr(3)); continue; }
        if (arg.find("--channels=") == 0) { embed_dim = std::stoi(arg.substr(11)); continue; }

        if (arg == "-b" || arg == "--batch") { if (i + 1 < argc) batch_size = std::stoi(argv[++i]); continue; }
        if (arg.find("-b=") == 0) { batch_size = std::stoi(arg.substr(3)); continue; }
        if (arg.find("--batch=") == 0) { batch_size = std::stoi(arg.substr(8)); continue; }

        if (arg == "-a" || arg == "--accumulate") { if (i + 1 < argc) grad_accum_steps = std::stoi(argv[++i]); continue; }
        if (arg.find("-a=") == 0) { grad_accum_steps = std::stoi(arg.substr(3)); continue; }
        if (arg.find("--accumulate=") == 0) { grad_accum_steps = std::stoi(arg.substr(13)); continue; }

        if (arg == "-w" || arg == "--window") { if (i + 1 < argc) max_seq_len = std::stoi(argv[++i]); continue; }
        if (arg.find("-w=") == 0) { max_seq_len = std::stoi(arg.substr(3)); continue; }
        if (arg.find("--window=") == 0) { max_seq_len = std::stoi(arg.substr(9)); continue; }

        if (arg == "-v" || arg == "--vocab") { if (i + 1 < argc) target_vocab_size = std::stoi(argv[++i]); continue; }
        if (arg.find("-v=") == 0) { target_vocab_size = std::stoi(arg.substr(3)); continue; }
        if (arg.find("--vocab=") == 0) { target_vocab_size = std::stoi(arg.substr(8)); continue; }

        if (arg == "-T" || arg == "--temp") { if (i + 1 < argc) temperature = std::stof(argv[++i]); continue; }
        if (arg.find("-T=") == 0) { temperature = std::stof(arg.substr(3)); continue; }
        if (arg.find("--temp=") == 0) { temperature = std::stof(arg.substr(7)); continue; }

        if (arg == "-K" || arg == "--topk") { if (i + 1 < argc) top_k = std::stoi(argv[++i]); continue; }
        if (arg.find("-K=") == 0) { top_k = std::stoi(arg.substr(3)); continue; }
        if (arg.find("--topk=") == 0) { top_k = std::stoi(arg.substr(7)); continue; }
    }

    strip_quotes(weights_path);
    strip_quotes(data_path);
    strip_quotes(prompt);

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
    std::cout << "[1/6] Using dataset at: " << data_path << "\n";

    // Tokenize and Build BPE Vocabulary Dynamically
    BytePairEncodingTokenizer tokenizer(target_vocab_size);
    tokenizer.build_vocab_from_file(data_path);
    int vocab_size = (int)tokenizer.get_vocab_size();
    std::cout << "[2/6] BPE Vocabulary Built! Target Vocab Size = " << target_vocab_size
              << " | Actual Vocab Size = " << vocab_size << " tokens.\n";

    std::vector<int> full_data = tokenizer.load_and_encode(data_path);
    std::cout << "      Total Dataset Tokens: " << full_data.size() << "\n";

    // Train / Validation Split
    std::vector<int> train_data, val_data;
    if (mode_train) {
        size_t split_idx = (size_t)(full_data.size() * 0.9);
        train_data.assign(full_data.begin(), full_data.begin() + split_idx);
        val_data.assign(full_data.begin() + split_idx, full_data.end());
        std::cout << "[3/6] Industry Standard Train/Val Split (90/10):\n"
                  << "      -> Training Tokens:   " << train_data.size() << "\n"
                  << "      -> Validation Tokens: " << val_data.size() << "\n\n";
    } else {
        std::cout << "[3/6] Inference Mode Selected (Skipping Train/Val Split)\n\n";
    }

    // Configure & Instantiate GPT Model Architecture
    GPTConfig config;
    config.vocab_size = vocab_size;
    config.max_seq_len = max_seq_len;
    config.embed_dim = embed_dim;
    config.num_layers = num_layers;
    config.use_gradient_checkpointing = use_checkpointing;

    GPT model(config);

    size_t total_params = 0;
    for (Tensor* param : model.get_parameters()) {
        total_params += param->size();
    }
    std::cout << "[4/6] Instantiated GPT Model Architecture:\n"
              << "      -> Vocab Size:    " << config.vocab_size << "\n"
              << "      -> Max Seq Len:   " << config.max_seq_len << "\n"
              << "      -> Embed Dim:     " << config.embed_dim << "\n"
              << "      -> Num Layers:    " << config.num_layers << "\n"
              << "      -> Checkpointing: " << (config.use_gradient_checkpointing ? "ENABLED (Block-Level Activation Recomputation)" : "DISABLED") << "\n"
              << "      -> Total Params:  " << total_params << " float32 parameters (~" 
              << (total_params * sizeof(float)) / 1024 << " KB)\n\n";

    // Training or Inference Execution
    std::mt19937 rng(42); // Seeded for reproducibility

    Device target_dev = use_gpu ? Device::CUDA : Device::CPU;
    if (use_gpu) {
#ifndef USE_CUDA
        std::cerr << "Error: --gpu flag specified, but MGPT was compiled without CUDA support (-DUSE_CUDA=OFF)!\n";
        std::cerr << "Please rebuild with CMake option -DUSE_CUDA=ON.\n";
        return -1;
#endif
        std::cout << "[4.5/6] Migrating GPT Model and Engine to CUDA GPU...\n";
        model.to(target_dev);
        model.init_scratchpad(256 * 1024 * 1024); // 256M floats (1 GB CUDA memory arena)
    }

    if (mode_infer_only) {
        std::cout << "[5/6] Loading Trained Model Weights from " << weights_path << "...\n";
        model.load_weights_bin(weights_path);
        if (use_gpu) model.to(target_dev);
    } else {
        float learning_rate = 1e-3f;
        int eval_interval = std::max(1, max_steps / 10);
        int print_interval = std::max(1, max_steps / 10);
        int eval_steps = 10;

        std::unique_ptr<Optimiser> optimizer = std::make_unique<AdamWOptimizer>(learning_rate);

        int* d_train_data = nullptr;
        int* d_val_data = nullptr;
        if (target_dev == Device::CUDA) {
            std::cout << "[4.8/6] Uploading Train and Validation Datasets to GPU Memory...\n";
            cuda_ops::allocate_int_memory(&d_train_data, train_data.size());
            cuda_ops::copy_int_host_to_device(d_train_data, train_data.data(), train_data.size());
            cuda_ops::allocate_int_memory(&d_val_data, val_data.size());
            cuda_ops::copy_int_host_to_device(d_val_data, val_data.data(), val_data.size());
        }

        Tensor x_batch({batch_size, config.max_seq_len}, 0.0f, target_dev);
        Tensor y_batch({batch_size, config.max_seq_len}, 0.0f, target_dev);
        Tensor logits({batch_size, config.max_seq_len, config.vocab_size}, 0.0f, target_dev);
        std::vector<Tensor*> params = model.get_parameters();

        std::cout << "[5/6] Starting Training Loop (AdamW, LR=" << learning_rate 
                  << ", Micro-Batch=" << batch_size
                  << ", Accum Steps=" << grad_accum_steps
                  << " [Effective Batch: " << (batch_size * grad_accum_steps) << "]"
                  << ", Steps=" << max_steps << ")...\n";
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
            return min_lr + 0.5f * (max_lr - min_lr) * (1.0f + std::cos(PI * decay_ratio));
        };

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int step = 1; step <= max_steps; ++step) {
            float current_lr = calculate_lr(step, max_steps, learning_rate, learning_rate * 0.1f);

            optimizer->set_lr(current_lr / (float)grad_accum_steps);

            auto step_start = std::chrono::high_resolution_clock::now();

            for (Tensor* param : params) {
                param->zero_grad();
            }

            float accum_train_loss = 0.0f;
            double total_fwd_ms = 0.0;
            double total_bwd_ms = 0.0;

            for (int micro = 0; micro < grad_accum_steps; ++micro) {
                get_batch(train_data, batch_size, config.max_seq_len, x_batch, y_batch, rng, d_train_data);

                auto t0 = std::chrono::high_resolution_clock::now();
                model.forward_into(x_batch, logits);
                auto t1 = std::chrono::high_resolution_clock::now();

                float micro_loss = model.compute_loss(logits, y_batch);
                auto t2 = std::chrono::high_resolution_clock::now();

                accum_train_loss += micro_loss;
                total_fwd_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                total_bwd_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
            }

            float train_loss = accum_train_loss / (float)grad_accum_steps;

            auto opt_start = std::chrono::high_resolution_clock::now();
            optimizer->step(params);
            auto opt_end = std::chrono::high_resolution_clock::now();
            double opt_ms = std::chrono::duration<double, std::milli>(opt_end - opt_start).count();

            auto step_end = std::chrono::high_resolution_clock::now();
            double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();

            if (step % print_interval == 0 || step == 1 || step == max_steps) {
                float val_loss = 0.0f;
                if (step % eval_interval == 0 || step == max_steps) {
                    val_loss = evaluate_loss(model, val_data, eval_steps, batch_size, config.max_seq_len, rng, target_dev, d_val_data);
                }

                int percent = (int)((step * 100.0) / max_steps);
                std::cout << "[" << std::setw(3) << step << "/" << max_steps << " (" << std::setw(3) << percent << "%)] "
                          << "LR:" << std::scientific << std::setprecision(2) << current_lr << " "
                          << "Fwd:" << (int)total_fwd_ms << "ms Loss:" << (int)total_bwd_ms << "ms Opt:" << (int)opt_ms << "ms "
                          << "| Step:" << (int)step_ms << "ms | Loss: " 
                          << std::fixed << std::setprecision(4) << train_loss;
                if (val_loss > 0.0f) {
                    std::cout << " | Val: " << val_loss;
                }
                std::cout << "\n" << std::flush;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_sec = std::chrono::duration<double>(end_time - start_time).count();
        std::cout << "------------------------------------------------------------\n";
        std::cout << "✅ Training Complete! Total Duration: " << std::fixed << std::setprecision(2) << total_sec << " seconds.\n\n";

        if (d_train_data) cuda_ops::free_int_memory(d_train_data);
        if (d_val_data) cuda_ops::free_int_memory(d_val_data);

        std::cout << "[6/6] Exporting Trained Model to " << weights_path << "...\n";
        model.save_weights_bin(weights_path);
    }

    std::cout << "\n--- 📜 Text Generation Sample (Prompt: \"" << prompt << "\" | Temp: " << temperature << " | Top-K: " << top_k << ") ---\n";
    std::string generated = generate_text(model, tokenizer, prompt, max_tokens, temperature, top_k, rng);
    std::cout << generated << "\n";
    std::cout << "------------------------------------------------------------\n\n";

    return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR]: " << e.what() << "\n";
        return -1;
    }
}
