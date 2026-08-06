#ifndef CLICONFIG_H
#define CLICONFIG_H

#include <string>
#include <iostream>

struct CLIConfig {
    bool mode_train = true;
    bool mode_infer_only = false;
    bool mode_resume = false;
    bool mode_virtual = false;
    bool use_gpu = false;
    bool use_checkpointing = true;
    bool use_flash_attention = true;

    std::string weights_path = "shakespeare_gpt.bin";
    std::string data_path = "input.txt";
    std::string vocab_path = "";
    std::string prompt = "To be or not to be";

    int max_tokens = 500;
    int max_steps = 1000;
    int start_step = -1;
    int total_steps = 0;
    int num_layers = 4;
    int embed_dim = 128;
    int batch_size = 16;
    int max_seq_len = 64;
    int target_vocab_size = 512;
    float temperature = 0.8f;
    int top_k = 15;
    float learning_rate = 3e-4f;
    int grad_accum_steps = 1;
    int num_experts = 8;
    int moe_top_k = 2;

    bool show_help = false;
};

inline void strip_quotes(std::string& s) {
    if (s.length() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
        s = s.substr(1, s.length() - 2);
    }
}

inline CLIConfig parse_arguments(int argc, char** argv) {
    CLIConfig config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: mgpt [options]\n"
                      << "Options:\n"
                      << "  -t, --train               Run in training mode (default)\n"
                      << "  -r, --resume              Resume training from existing checkpoint weights & step counter\n"
                      << "  -i, --infer               Run in inference/generation mode (load weights without training)\n"
                      << "  -v, --virtual             Run a dry-run memory profiler to output max VRAM requirements\n"
                      << "  -g, --gpu                 Run training and inference using CUDA GPU engine\n"
                      << "  -k, --checkpointing       Enable gradient/activation checkpointing (default: enabled)\n"
                      << "      --no-checkpointing    Disable gradient/activation checkpointing\n"
                      << "      --flash-attention     Enable custom Flash Attention scalar kernels (default: enabled)\n"
                      << "      --no-flash-attention  Disable Flash Attention and fallback to standard cuBLAS GEMM\n"
                      << "  -f, --file <path>         Path to model weights file (.bin) (default: shakespeare_gpt.bin)\n"
                      << "  -d, --data <path>         Path to training dataset (default: input.txt)\n"
                      << "      --vocab-file <path>   Path to master .vocab.bin dictionary file for consistent sharded training\n"
                      << "  -v, --vocab <int>         Target BPE vocabulary size (default: 512)\n"
                      << "  -T, --temp <float>        Sampling temperature for text generation (default: 0.8)\n"
                      << "  -K, --topk <int>          Top-K nucleus filtering for generation (default: 15)\n"
                      << "  -p, --prompt <str>        Text generation prompt (default: \"To be or not to be\")\n"
                      << "  -n, --tokens <int>        Number of characters to generate (default: 500)\n"
                      << "  -s, --steps <int>         Number of training steps on current shard (default: 1000)\n"
                      << "      --start-step <int>    Explicitly override starting step counter when resuming\n"
                      << "      --total-steps <int>   Global total steps across all shards for continuous Cosine LR decay\n"
                      << "  -l, --layers <int>        Number of Transformer blocks (default: 4)\n"
                      << "  -c, --channels <int>      Embedding channel dimension (default: 128)\n"
                      << "  -b, --batch <int>         Training batch size (default: 16)\n"
                      << "  -w, --window <int>        Context window / sequence length (default: 64)\n"
                      << "  -a, --accumulate <int>    Gradient accumulation steps (default: 1)\n"
                      << "  -e, -lr, --lr <float>     Initial/peak learning rate (default: 0.0003)\n"
                      << "      --experts <int>       Number of MoE experts (default: 8, 0 to disable)\n"
                      << "      --moe-topk <int>      Number of MoE experts to route to per token (default: 2)\n"
                      << "  -h, --help                Show this help message and exit\n\n"
                      << "Examples:\n"
                      << "  mgpt -t -g -v=256 -f=\"my_model.bin\" -s=2000 -l=6 -b=64 -w=128\n"
                      << "  mgpt -t -g --resume -d=\"shard_01.txt\" --vocab-file=\"shard_00.txt.vocab.bin\" -f=\"my_model.bin\" -s=2000 --total-steps=40000\n"
                      << "  mgpt -i -g -T=0.7 -f=\"my_model.bin\" -p=\"O Romeo, Romeo!\" -n=1000\n";
            config.show_help = true;
            return config;
        }
        if (arg == "-t" || arg == "--train") { config.mode_train = true; config.mode_infer_only = false; config.mode_virtual = false; continue; }
        if (arg == "-r" || arg == "--resume") { config.mode_resume = true; config.mode_train = true; config.mode_infer_only = false; config.mode_virtual = false; continue; }
        if (arg == "-i" || arg == "--infer") { config.mode_infer_only = true; config.mode_train = false; config.mode_virtual = false; continue; }
        if (arg == "--virtual") { config.mode_virtual = true; config.mode_train = false; config.mode_infer_only = false; continue; }
        if (arg == "-g" || arg == "--gpu") { config.use_gpu = true; continue; }
        if (arg == "-k" || arg == "--checkpointing") { config.use_checkpointing = true; continue; }
        if (arg == "--no-checkpointing") { config.use_checkpointing = false; continue; }
        if (arg == "--flash-attention") { config.use_flash_attention = true; continue; }
        if (arg == "--no-flash-attention") { config.use_flash_attention = false; continue; }

        if (arg == "-f" || arg == "--file") { if (i + 1 < argc) config.weights_path = argv[++i]; continue; }
        if (arg.find("-f=") == 0) { config.weights_path = arg.substr(3); continue; }
        if (arg.find("--file=") == 0) { config.weights_path = arg.substr(7); continue; }

        if (arg == "-d" || arg == "--data") { if (i + 1 < argc) config.data_path = argv[++i]; continue; }
        if (arg.find("-d=") == 0) { config.data_path = arg.substr(3); continue; }
        if (arg.find("--data=") == 0) { config.data_path = arg.substr(7); continue; }

        if (arg == "--vocab-file") { if (i + 1 < argc) config.vocab_path = argv[++i]; continue; }
        if (arg.find("--vocab-file=") == 0) { config.vocab_path = arg.substr(13); continue; }

        if (arg == "-p" || arg == "--prompt") { if (i + 1 < argc) config.prompt = argv[++i]; continue; }
        if (arg.find("-p=") == 0) { config.prompt = arg.substr(3); continue; }
        if (arg.find("--prompt=") == 0) { config.prompt = arg.substr(9); continue; }

        if (arg == "-n" || arg == "--tokens") { if (i + 1 < argc) config.max_tokens = std::stoi(argv[++i]); continue; }
        if (arg.find("-n=") == 0) { config.max_tokens = std::stoi(arg.substr(3)); continue; }
        if (arg.find("--tokens=") == 0) { config.max_tokens = std::stoi(arg.substr(9)); continue; }

        if (arg == "-s" || arg == "--steps") { if (i + 1 < argc) config.max_steps = std::stoi(argv[++i]); continue; }
        if (arg.find("-s=") == 0) { config.max_steps = std::stoi(arg.substr(3)); continue; }
        if (arg.find("--steps=") == 0) { config.max_steps = std::stoi(arg.substr(8)); continue; }

        if (arg == "--start-step") { if (i + 1 < argc) config.start_step = std::stoi(argv[++i]); continue; }
        if (arg.find("--start-step=") == 0) { config.start_step = std::stoi(arg.substr(13)); continue; }

        if (arg == "--total-steps") { if (i + 1 < argc) config.total_steps = std::stoi(argv[++i]); continue; }
        if (arg.find("--total-steps=") == 0) { config.total_steps = std::stoi(arg.substr(14)); continue; }

        if (arg == "-l" || arg == "--layers") { if (i + 1 < argc) config.num_layers = std::stoi(argv[++i]); continue; }
        if (arg.find("-l=") == 0) { config.num_layers = std::stoi(arg.substr(3)); continue; }
        if (arg.find("--layers=") == 0) { config.num_layers = std::stoi(arg.substr(9)); continue; }

        if (arg == "-c" || arg == "--channels") { if (i + 1 < argc) config.embed_dim = std::stoi(argv[++i]); continue; }
        if (arg.find("-c=") == 0) { config.embed_dim = std::stoi(arg.substr(3)); continue; }
        if (arg.find("--channels=") == 0) { config.embed_dim = std::stoi(arg.substr(11)); continue; }

        if (arg == "-b" || arg == "--batch") { if (i + 1 < argc) config.batch_size = std::stoi(argv[++i]); continue; }
        if (arg.find("-b=") == 0) { config.batch_size = std::stoi(arg.substr(3)); continue; }
        if (arg.find("--batch=") == 0) { config.batch_size = std::stoi(arg.substr(8)); continue; }

        if (arg == "-a" || arg == "--accumulate") { if (i + 1 < argc) config.grad_accum_steps = std::stoi(argv[++i]); continue; }
        if (arg.find("-a=") == 0) { config.grad_accum_steps = std::stoi(arg.substr(3)); continue; }
        if (arg.find("--accumulate=") == 0) { config.grad_accum_steps = std::stoi(arg.substr(13)); continue; }

        if (arg == "-w" || arg == "--window") { if (i + 1 < argc) config.max_seq_len = std::stoi(argv[++i]); continue; }
        if (arg.find("-w=") == 0) { config.max_seq_len = std::stoi(arg.substr(3)); continue; }
        if (arg.find("--window=") == 0) { config.max_seq_len = std::stoi(arg.substr(9)); continue; }

        if (arg.find("--window=") == 0) { config.max_seq_len = std::stoi(arg.substr(9)); continue; }

        if (arg == "-v" || arg == "--vocab") { if (i + 1 < argc) config.target_vocab_size = std::stoi(argv[++i]); continue; }
        if (arg.find("-v=") == 0) { config.target_vocab_size = std::stoi(arg.substr(3)); continue; }
        if (arg.find("--vocab=") == 0) { config.target_vocab_size = std::stoi(arg.substr(8)); continue; }

        if (arg == "-T" || arg == "--temp") { if (i + 1 < argc) config.temperature = std::stof(argv[++i]); continue; }
        if (arg.find("-T=") == 0) { config.temperature = std::stof(arg.substr(3)); continue; }
        if (arg.find("--temp=") == 0) { config.temperature = std::stof(arg.substr(7)); continue; }

        if (arg == "-K" || arg == "--topk") { if (i + 1 < argc) config.top_k = std::stoi(argv[++i]); continue; }
        if (arg.find("-K=") == 0) { config.top_k = std::stoi(arg.substr(3)); continue; }
        if (arg.find("--topk=") == 0) { config.top_k = std::stoi(arg.substr(7)); continue; }

        if (arg == "-lr" || arg == "--lr" || arg == "-e") { if (i + 1 < argc) config.learning_rate = std::stof(argv[++i]); continue; }
        if (arg.find("-lr=") == 0) { config.learning_rate = std::stof(arg.substr(4)); continue; }
        if (arg.find("--lr=") == 0) { config.learning_rate = std::stof(arg.substr(5)); continue; }
        if (arg.find("-e=") == 0) { config.learning_rate = std::stof(arg.substr(3)); continue; }

        if (arg == "--experts") { if (i + 1 < argc) config.num_experts = std::stoi(argv[++i]); continue; }
        if (arg.find("--experts=") == 0) { config.num_experts = std::stoi(arg.substr(10)); continue; }

        if (arg == "--moe-topk") { if (i + 1 < argc) config.moe_top_k = std::stoi(argv[++i]); continue; }
        if (arg.find("--moe-topk=") == 0) { config.moe_top_k = std::stoi(arg.substr(11)); continue; }
    }

    strip_quotes(config.weights_path);
    strip_quotes(config.data_path);
    strip_quotes(config.vocab_path);
    strip_quotes(config.prompt);

    return config;
}

#endif // CLICONFIG_H
