# C++ & CUDA Transformer Engine

A zero-allocation autoregressive Transformer/GPT training and inference engine built from scratch in modern C++ and hardware-accelerated CUDA. Designed with zero external dependencies—pure linear algebra, custom autograd, memory scratchpad recycling, and modern LLM architectural features.

> **📖 Systems Engineering Diary**: For a complete chronological log of architectural decisions, performance debugging, VRAM optimizations, and mathematical discoveries during the building of this engine, please read the [diary.md](diary.md).

## Features

### Architecture
- **SwiGLU Activation Network**: Implements modern gated SwiGLU (`Swish(Gate(X)) ⊙ Up(X) -> Down`) MLP layers for enhanced representational capacity and faster convergence.
- **Root Mean Square Normalization (RMSNorm)**: Lean, stable pre-attention and pre-MLP normalization (`RMSNormLayer`) with learnable $\gamma$ scale parameter.
- **Causal Multi-Head Self-Attention (MHA)**: Full multi-head causal attention with query/key/value projection slicing and multi-head concatenation kernels.
- **Byte Pair Encoding (BPE) & Character Tokenizers**: Built-in BPE tokenizer (`BytePairEncodingTokenizer`) with configurable vocabulary size (`--vocab`), alongside a polymorphic `Tokenizer` interface supporting raw character-level tokenization.

### Memory & Hardware Acceleration
- **Custom Flash Attention**: Native CUDA implementations of Flash Attention kernels (`flash_attention_forward` and `flash_attention_backward`) to eliminate the $O(T^2)$ memory bottleneck, significantly reducing VRAM footprint for long context windows.
- **Zero-Allocation GPU Scratchpad**: Reuses pre-allocated temporary memory buffers (`Scratchpad`) across forward and backward passes to eliminate runtime CUDA memory allocations.
- **Activation / Gradient Checkpointing**: Recomputes intermediate layer activations during backpropagation (`-k` / `--checkpointing`), drastically reducing peak GPU memory usage for deep models.
- **Gradient Accumulation**: Supports multi-step micro-batch gradient accumulation (`-a` / `--accumulate`) to train with massive effective batch sizes on consumer GPUs.
- **Dual CUDA & CPU Backends**: Highly optimized CUDA kernels (`cuda_ops.cu`) with clean CPU C++ fallback stubs (`#else`) allowing builds and execution on any platform.
- **AdamW Optimizer**: Built-in fused AdamW optimizer with decoupled weight decay and momentum tracking.

---

##  Building & Compiling (Windows / PowerShell)

### Prerequisites
1. **Microsoft Visual Studio Community 2022** (with *"Desktop development with C++"* workload).
2. **NVIDIA CUDA Toolkit** (v12.x or v13.x installed *after* Visual Studio).
3. **CMake** 3.20+.


### Complete CLI Flags & Options

| Short Flag | Long Flag | Description | Default Value |
| :---: | :--- | :--- | :---: |
| `-t` | `--train` | Run in **Training Mode** (trains model & saves checkpoint) | *Default* |
| `-r` | `--resume` | Resume training from existing checkpoint weights & step progress header | `false` |
| `-i` | `--infer` | Run in **Inference Mode** (loads checkpoint & generates text) | `false` |
| `-g` | `--gpu` | Enable **CUDA GPU Acceleration** for training & generation | `false` |
| `-k` | `--checkpointing` | Enable **Activation/Gradient Checkpointing** for memory efficiency | `true` |
| | `--no-checkpointing` | Disable activation checkpointing (store all activations) | `false` |
| | `--flash-attention` | Enable memory-efficient custom Flash Attention kernels | `true` |
| | `--no-flash-attention`| Disable Flash Attention (fall back to cuBLAS scalar matrices) | `false` |
| `-f` | `--file <path>` | Path to save/load model binary checkpoint (`.bin`) | `"shakespeare_gpt.bin"` |
| `-d` | `--data <path>` | Path to training text corpus shard/file | `"input.txt"` |
| | `--vocab-file <path>` | Path to master `.vocab.bin` dictionary for consistent multi-shard training | |
| `-v` | `--vocab <int>` | Target BPE vocabulary size | `512` |
| `-p` | `--prompt <str>` | Initial prompt string for autoregressive text generation | `"To be or not to be"` |
| `-n` | `--tokens <int>` | Number of tokens/characters to generate | `500` |
| `-s` | `--steps <int>` | Number of training optimization steps on current shard | `1000` |
| | `--start-step <int>` | Explicitly override starting step counter when resuming | `0` |
| | `--total-steps <int>` | Global total steps across all shards for continuous Cosine LR decay | `0` |
| `-l` | `--layers <int>` | Number of Transformer blocks ($L$) | `4` |
| `-c` | `--channels <int>` | Embedding channel dimension / hidden size ($C$) | `128` |
| `-b` | `--batch <int>` | Micro-batch size per forward/backward pass ($B$) | `16` |
| `-w` | `--window <int>` | Context window / sequence length ($T$) | `64` |
| `-a` | `--accumulate <int>` | Number of gradient accumulation steps | `1` |
| `-e` | `--lr <float>` | Initial / peak learning rate for AdamW optimizer | `0.0003` |
| `-T` | `--temp <float>` | Sampling temperature for text generation | `0.8` |
| `-K` | `--topk <int>` | Top-K nucleus filtering for generation | `15` |
| `-h` | `--help` | Display usage summary and exit | |

---