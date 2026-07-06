#include "GPT.h"

GPT::GPT(const GPTConfig& config)
    : config(config),
      tok_emb(config.vocab_size, config.embed_dim),
      pos_emb(config.max_seq_len, config.embed_dim),
      ln_f(config.embed_dim),
      lm_head(config.embed_dim, config.vocab_size) {
    for (int i = 0; i < config.num_layers; ++i) {
        blocks.push_back(std::make_unique<TransformerBlock>(config.embed_dim));
    }
}

// FORWARD PASS: Token + Pos Embeddings -> L Transformer Blocks -> LayerNorm -> LM Head
Tensor GPT::forward(const Tensor& input_ids) {
    if (input_ids.shape.size() != 2) {
        std::cerr << "GPT::forward: input_ids must be 2D {Batch, Time}!" << std::endl;
        exit(-1);
    }
    int B = input_ids.shape[0];
    int T = input_ids.shape[1];
    if (T > config.max_seq_len) {
        std::cerr << "GPT::forward: sequence length T (" << T 
                  << ") exceeds max_seq_len (" << config.max_seq_len << ")!" << std::endl;
        exit(-1);
    }
    cached_input_ids = input_ids;

    // Token Embeddings: {B, T} -> {B, T, C}
    cached_tok_emb = tok_emb.forward(input_ids);

    // Positional Embeddings: generate [0, 1, ..., T-1] across batch -> {B, T, C}
    Tensor pos_ids({B, T}, 0.0f);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            pos_ids.data[b * T + t] = (float)t;
        }
    }
    cached_pos_emb = pos_emb.forward(pos_ids);

    // Combine embeddings
    cached_x0 = cached_tok_emb + cached_pos_emb;

    // Pass sequentially through Transformer Blocks
    Tensor curr_x = cached_x0;
    for (size_t i = 0; i < blocks.size(); ++i) {
        curr_x = blocks[i]->forward(curr_x);
    }

    // Final Layer Normalization & Language Model Head Projection
    cached_ln_f_out = ln_f.forward(curr_x);
    cached_logits = lm_head.forward(cached_ln_f_out); // Shape {B, T, VocabSize}

    return cached_logits;
}

// LOSS EVALUATION: Computes Cross-Entropy loss and triggers backprop
float GPT::compute_loss(const Tensor& logits, const Tensor& target_ids) {
    float loss = loss_layer.forward_loss(logits, target_ids);
    Tensor d_logits = loss_layer.backward(Tensor()); // Generates initial (P - 1) / (B * T) grad
    backward(d_logits);
    return loss;
}

// BACKWARD PASS: Chain rule backwards through LM Head, LayerNorm, Blocks, and Embeddings
Tensor GPT::backward(const Tensor& d_logits) {
    // Backprop through LM Head
    Tensor d_ln_f = lm_head.backward(d_logits);

    // Backprop through Final LayerNorm
    Tensor d_curr = ln_f.backward(d_ln_f);

    // Backprop through Transformer Blocks in reverse order
    for (int i = (int)blocks.size() - 1; i >= 0; --i) {
        d_curr = blocks[i]->backward(d_curr);
    }

    // Backprop through Embeddings (sum of tok and pos branches)
    tok_emb.backward(d_curr);
    pos_emb.backward(d_curr);

    return Tensor(); // Return empty tensor for discrete integer inputs
}

std::vector<Tensor*> GPT::get_parameters() {
    std::vector<Tensor*> params;
    auto p_tok = tok_emb.get_parameters(); params.insert(params.end(), p_tok.begin(), p_tok.end());
    auto p_pos = pos_emb.get_parameters(); params.insert(params.end(), p_pos.begin(), p_pos.end());
    for (auto& block : blocks) {
        auto p_b = block->get_parameters();
        params.insert(params.end(), p_b.begin(), p_b.end());
    }
    auto p_ln = ln_f.get_parameters(); params.insert(params.end(), p_ln.begin(), p_ln.end());
    auto p_head = lm_head.get_parameters(); params.insert(params.end(), p_head.begin(), p_head.end());
    return params;
}

// Save & Load raw float weights to standard .bin format
void GPT::save_weights_bin(const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "GPT::save_weights_bin: Could not open file " << filepath << " for writing!\n";
        exit(-1);
    }
    // Write 4 integer header values
    out.write((char*)&config.vocab_size, sizeof(int));
    out.write((char*)&config.max_seq_len, sizeof(int));
    out.write((char*)&config.embed_dim, sizeof(int));
    out.write((char*)&config.num_layers, sizeof(int));

    // Write all parameter vectors sequentially
    for (Tensor* param : get_parameters()) {
        out.write((char*)param->data.data(), param->size() * sizeof(float));
    }
    out.close();
    std::cout << "Successfully saved GPT model weights (" << get_parameters().size() 
              << " parameter tensors) to " << filepath << "!\n";
}

void GPT::load_weights_bin(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "GPT::load_weights_bin: Could not open file " << filepath << " for reading!\n";
        exit(-1);
    }
    // Read and verify compatibility header
    int v_size = 0, seq_len = 0, e_dim = 0, n_layers = 0;
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

    // Read all parameter vectors sequentially
    for (Tensor* param : get_parameters()) {
        in.read((char*)param->data.data(), param->size() * sizeof(float));
    }
    in.close();
    std::cout << "Successfully loaded GPT model weights from " << filepath << "!\n";
}
