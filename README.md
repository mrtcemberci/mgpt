# C++ & CUDA Transformer Engine

A state-of-the-art, zero-allocation autoregressive Transformer/GPT training and inference engine built from scratch in modern C++ and hardware-accelerated CUDA. Designed with zero external dependencies—pure linear algebra, custom autograd, memory scratchpad recycling, and modern LLM architectural features.

---

## Features

### Architecture
- **SwiGLU Activation Network**: Implements modern gated SwiGLU (`Swish(Gate(X)) ⊙ Up(X) -> Down`) MLP layers for enhanced representational capacity and faster convergence.
- **Root Mean Square Normalization (RMSNorm)**: Lean, stable pre-attention and pre-MLP normalization (`RMSNormLayer`) with learnable $\gamma$ scale parameter.
- **Causal Multi-Head Self-Attention (MHA)**: Full multi-head causal attention with query/key/value projection slicing and multi-head concatenation kernels.
- **Byte Pair Encoding (BPE) & Character Tokenizers**: Built-in BPE tokenizer (`BytePairEncodingTokenizer`) with configurable vocabulary size (`--vocab`), alongside a polymorphic `Tokenizer` interface supporting raw character-level tokenization.

### Memory & Hardware Acceleration
- **Zero-Allocation GPU Scratchpad**: Reuses pre-allocated temporary memory buffers (`Scratchpad`) across forward and backward passes to eliminate runtime CUDA memory allocations.
- **Activation / Gradient Checkpointing**: Recomputes intermediate layer activations during backpropagation (`-k` / `--checkpointing`), drastically reducing peak GPU memory usage for deep models.
- **Gradient Accumulation**: Supports multi-step micro-batch gradient accumulation (`-a` / `--accumulate`) to train with massive effective batch sizes on consumer GPUs.
- **Dual CUDA & CPU Backends**: Highly optimized CUDA kernels (`cuda_ops.cu`) with clean CPU C++ fallback stubs (`#else`) allowing builds and execution on any platform.
- **AdamW Optimizer**: Built-in fused AdamW optimizer with decoupled weight decay and momentum tracking.

---

## 🛠️ Building & Compiling (Windows / PowerShell)

### Prerequisites
1. **Microsoft Visual Studio Community 2022** (with *"Desktop development with C++"* workload).
2. **NVIDIA CUDA Toolkit** (v12.x or v13.x installed *after* Visual Studio).
3. **CMake** 3.20+.

### Build Command
Open PowerShell in the root directory of the repository and compile using CMake:

```powershell
# Set your CUDA Toolkit directory (adjust v13.3 to match your installed version, e.g. v12.6)
$env:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\"

# Configure and compile using CMake in Release mode
cmake --build build_release --config Release --target mgpt

# Add CUDA binaries to your PowerShell PATH
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64;" + $env:PATH
```

Compiled binary output: `.\build_release\Release\mgpt.exe`

---

## 💻 Command-Line Usage

```bash
mgpt [options]
```

### Complete CLI Flags & Options

| Short Flag | Long Flag | Description | Default Value |
| :---: | :--- | :--- | :---: |
| `-t` | `--train` | Run in **Training Mode** (trains model & saves checkpoint) | *Default* |
| `-r` | `--resume` | Resume training from existing checkpoint weights & step progress header | `false` |
| `-i` | `--infer` | Run in **Inference Mode** (loads checkpoint & generates text) | `false` |
| `-g` | `--gpu` | Enable **CUDA GPU Acceleration** for training & generation | `false` |
| `-k` | `--checkpointing` | Enable **Activation/Gradient Checkpointing** for memory efficiency | `true` |
| | `--no-checkpointing` | Disable activation checkpointing (store all activations) | `false` |
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
| `-T` | `--temp <float>` | Sampling temperature for text generation | `0.8` |
| `-K` | `--topk <int>` | Top-K nucleus filtering for generation | `15` |
| `-h` | `--help` | Display usage summary and exit | |

---

## 🔥 Ready-to-Run Examples

### 1. Multi-Shard Large Dataset Training Workflow (e.g. TinyStories)
Train sequentially across dataset shards (`tinystories/shard_00.txt`, `tinystories/shard_01.txt`, ...) while sharing a master BPE vocabulary (`--vocab-file`) and continuous Cosine LR schedule (`--total-steps`):

#### Manual Shard-by-Shard Execution
```powershell
# Shard 00: Builds master vocabulary & trains steps 1 -> 3000 (out of 72,000 total)
.\build_release\Release\mgpt.exe -t -g -v=4096 -d="tinystories/shard_00.txt" -f="tinystories.bin" -s=3000 --total-steps=72000 -l=12 -c=384 -w=256 -b=16 -a=2

# Shard 01: Resumes from step 3000 -> 6000 and encodes shard using master vocabulary
.\build_release\Release\mgpt.exe -t -g --resume -d="tinystories/shard_01.txt" --vocab-file="tinystories/shard_00.txt.vocab.bin" -f="tinystories.bin" -s=3000 --total-steps=72000 -l=12 -c=384 -w=256 -b=16 -a=2
```

#### Automated Overnight Pipeline (PowerShell Loop)
```powershell
# 1. Start Shard 00 (initializes weights and creates master vocabulary)
.\build_release\Release\mgpt.exe -t -g -v=4096 -d="tinystories/shard_00.txt" -f="tinystories.bin" -s=3000 --total-steps=72000 -l=12 -c=384 -w=256 -b=16 -a=2

# 2. Automatically loop through Shards 01 to 23
1..23 | ForEach-Object {
    $idx = "{0:D2}" -f $_
    Write-Host "=== Resuming Training on Shard $idx ===" -ForegroundColor Cyan
    .\build_release\Release\mgpt.exe -t -g --resume -d="tinystories/shard_${idx}.txt" --vocab-file="tinystories/shard_00.txt.vocab.bin" -f="tinystories.bin" -s=3000 --total-steps=72000 -l=12 -c=384 -w=256 -b=16 -a=2
}
```