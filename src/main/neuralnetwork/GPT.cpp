#include "GPT.h"
#include "cuda_ops.h"

GPT::GPT(const GPTConfig& config)
    : config(config),
      tok_emb(config.vocab_size, config.embed_dim),
      ln_f(config.embed_dim),
      lm_head(config.embed_dim, config.vocab_size) {
    for (int i = 0; i < config.num_layers; ++i) {
        blocks.push_back(std::make_unique<TransformerBlock>(config.embed_dim, config.num_heads, config.num_experts, config.top_k, config.use_flash_attention));
    }
}

// FORWARD PASS: Token Embeddings -> L Transformer Blocks -> LayerNorm -> LM Head
void GPT::forward_into(const Tensor& input_ids, Tensor& output) {
    if (scratchpad) {
        global_savepoint = scratchpad->get_savepoint();
    }

    if (input_ids.shape.size() != 2) {
        std::cerr << "GPT::forward_into: input_ids must be 2D {Batch, Time}!" << std::endl;
        exit(-1);
    }
    int B = input_ids.shape[0];
    int T = input_ids.shape[1];
    if (T > config.max_seq_len) {
        std::cerr << "GPT::forward_into: sequence length T (" << T 
                  << ") exceeds max_seq_len (" << config.max_seq_len << ")!" << std::endl;
        exit(-1);
    }
    cached_input_ids = input_ids;

    // Token Embeddings: {B, T} -> {B, T, C}
    tok_emb.forward_into(input_ids, cached_tok_emb);
    
    // Pass sequentially through Transformer Blocks
    Tensor curr_x = cached_tok_emb;
    for (size_t i = 0; i < blocks.size(); ++i) {
        size_t block_savepoint = 0;
        if (config.use_gradient_checkpointing && scratchpad) {
            block_savepoint = scratchpad->get_savepoint();
        }

        blocks[i]->forward_into(curr_x, blocks[i]->cached_out);
        curr_x = blocks[i]->cached_out;

        if (config.use_gradient_checkpointing && scratchpad) {
            scratchpad->restore_savepoint(block_savepoint);
        }
    }

    // Final Layer Normalization & Language Model Head Projection
    ln_f.forward_into(curr_x, cached_ln_f_out);
    lm_head.forward_into(cached_ln_f_out, output); // Shape {B, T, VocabSize}
    cached_logits = output;
}

Tensor GPT::forward(const Tensor& input_ids) {
    forward_into(input_ids, cached_logits);
    return cached_logits;
}

// LOSS EVALUATION: Computes Cross-Entropy loss and triggers backprop
float GPT::compute_loss(const Tensor& logits, const Tensor& target_ids) {
    float loss = loss_layer.forward_loss(logits, target_ids);
    loss_layer.backward_into(Tensor(), cached_d_logits); // Generates initial (P - 1) / (B * T) grad
    backward_into(cached_d_logits, cached_dummy_din);
    return loss;
}

// BACKWARD PASS: Chain rule backwards through LM Head, LayerNorm, Blocks, and Embeddings
void GPT::backward_into(const Tensor& d_logits, Tensor& din) {
    // Backprop through LM Head
    lm_head.backward_into(d_logits, cached_d_ln_f);

    // Backprop through Final LayerNorm
    ln_f.backward_into(cached_d_ln_f, cached_d_curr);

    // Backprop through Transformer Blocks in reverse order
    if (cached_d_blocks.size() != blocks.size()) {
        cached_d_blocks.resize(blocks.size());
    }
    for (int i = (int)blocks.size() - 1; i >= 0; --i) {
        size_t block_savepoint = 0;
        if (config.use_gradient_checkpointing && scratchpad) {
            block_savepoint = scratchpad->get_savepoint();
            // Recompute Block i's forward activations from its saved input
            blocks[i]->forward_into(blocks[i]->cached_input, blocks[i]->cached_out);
        }

        if (i == (int)blocks.size() - 1) {
            blocks[i]->backward_into(cached_d_curr, cached_d_blocks[i]);
        } else {
            blocks[i]->backward_into(cached_d_blocks[i+1], cached_d_blocks[i]);
        }

        if (config.use_gradient_checkpointing && scratchpad) {
            scratchpad->restore_savepoint(block_savepoint);
        }
    }

    // Backprop through Embeddings (tok branch only since pos_emb is replaced by RoPE)
    if (!blocks.empty()) {
        tok_emb.backward_into(cached_d_blocks[0], cached_dummy_din);
    } else {
        tok_emb.backward_into(cached_d_curr, cached_dummy_din);
    }
    cached_dX = Tensor();

    if (scratchpad) {
        scratchpad->restore_savepoint(global_savepoint);
    }
}

Tensor GPT::backward(const Tensor& d_logits) {
    backward_into(d_logits, cached_dX);
    return Tensor(); // Return empty tensor for discrete integer inputs
}

std::vector<Tensor*> GPT::get_parameters() {
    std::vector<Tensor*> params;
    auto p_tok = tok_emb.get_parameters(); params.insert(params.end(), p_tok.begin(), p_tok.end());
    for (auto& block : blocks) {
        auto p_b = block->get_parameters();
        params.insert(params.end(), p_b.begin(), p_b.end());
    }
    auto p_ln = ln_f.get_parameters(); params.insert(params.end(), p_ln.begin(), p_ln.end());
    auto p_head = lm_head.get_parameters(); params.insert(params.end(), p_head.begin(), p_head.end());
    return params;
}

// Save & Load raw float weights to standard .bin format with step progress header
void GPT::save_weights_bin(const std::string& filepath, int completed_steps) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "GPT::save_weights_bin: Could not open file " << filepath << " for writing!\n";
        exit(-1);
    }
    // Write 5 integer header values (including completed_steps)
    out.write((char*)&config.vocab_size, sizeof(int));
    out.write((char*)&config.max_seq_len, sizeof(int));
    out.write((char*)&config.embed_dim, sizeof(int));
    out.write((char*)&config.num_layers, sizeof(int));
    out.write((char*)&completed_steps, sizeof(int));

    // Write all parameter vectors sequentially
    for (Tensor* param : get_parameters()) {
        if (param->device == Device::CUDA) {
            std::vector<float> host_tmp(param->size());
            cuda_ops::copy_device_to_host(host_tmp.data(), param->get_data_ptr(), param->size());
            out.write((char*)host_tmp.data(), param->size() * sizeof(float));
        } else {
            out.write((char*)param->data.data(), param->size() * sizeof(float));
        }
    }
    out.close();
    std::cout << "Successfully saved GPT model weights (" << get_parameters().size() 
              << " parameter tensors, step " << completed_steps << ") to " << filepath << "!\n";
}

int GPT::load_weights_bin(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "GPT::load_weights_bin: Could not open file " << filepath << " for reading!\n";
        exit(-1);
    }
    in.seekg(0, std::ios::end);
    size_t file_bytes = in.tellg();
    in.seekg(0, std::ios::beg);

    // Read and verify compatibility header
    int v_size = 0, seq_len = 0, e_dim = 0, n_layers = 0, completed_steps = 0;
    in.read((char*)&v_size, sizeof(int));
    in.read((char*)&seq_len, sizeof(int));
    in.read((char*)&e_dim, sizeof(int));
    in.read((char*)&n_layers, sizeof(int));

    if (v_size != config.vocab_size || seq_len != config.max_seq_len || 
        e_dim != config.embed_dim || n_layers != config.num_layers) {
        std::cerr << "GPT::load_weights_bin: Model architecture mismatch in binary header!\n"
                  << "  Expected: (" << config.vocab_size << ", " << config.max_seq_len << ", " 
                  << config.embed_dim << ", " << config.num_layers << ")\n"
                  << "  Found:    (" << v_size << ", " << seq_len << ", " << e_dim << ", " << n_layers << ")\n";
        exit(-1);
    }

    size_t expected_param_bytes = 0;
    for (Tensor* param : get_parameters()) {
        expected_param_bytes += param->size() * sizeof(float);
    }

    // Check if 5th header integer (completed_steps) exists in file
    if (file_bytes >= 5 * sizeof(int) + expected_param_bytes) {
        in.read((char*)&completed_steps, sizeof(int));
    }

    // Read all parameter vectors sequentially
    for (Tensor* param : get_parameters()) {
        if (param->device == Device::CUDA) {
            std::vector<float> host_tmp(param->size());
            in.read((char*)host_tmp.data(), param->size() * sizeof(float));
            cuda_ops::copy_host_to_device(param->get_data_ptr(), host_tmp.data(), param->size());
        } else {
            in.read((char*)param->data.data(), param->size() * sizeof(float));
        }
    }
    in.close();
    std::cout << "Successfully loaded GPT model weights from " << filepath 
              << " (restored at completed step " << completed_steps << ")!\n";
    return completed_steps;
}

void GPT::set_scratchpad(Scratchpad* pad) {
    this->scratchpad = pad;
    tok_emb.set_scratchpad(pad);
    for (auto& block : blocks) {
        if (block) block->set_scratchpad(pad);
    }
    ln_f.set_scratchpad(pad);
    lm_head.set_scratchpad(pad);
    loss_layer.set_scratchpad(pad);
}

void GPT::init_scratchpad(size_t capacity_floats) {
    owned_scratchpad = std::make_unique<Scratchpad>(capacity_floats, ScratchpadDevice::CUDA);
    set_scratchpad(owned_scratchpad.get());
}

ScratchpadFootprint GPT::get_footprint(int B, int T) {
    size_t C = config.embed_dim;
    size_t fwd_standing = (size_t)(B * T * C + 2 * B * T * config.vocab_size);
    size_t block_fwd = 0, block_bwd = 0;
    if (!blocks.empty()) {
        block_fwd = blocks[0]->get_footprint(B, T).fwd_peak();
        block_bwd = blocks[0]->get_footprint(B, T).bwd_peak();
    }
    size_t bwd_temp = (config.use_gradient_checkpointing ? block_bwd : (config.num_layers * (!blocks.empty() ? blocks[0]->get_footprint(B,T).fwd_standing : 0) + block_bwd));
    return {fwd_standing, block_fwd, bwd_temp};
}
