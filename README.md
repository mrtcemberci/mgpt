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
| `-e` | `--lr <float>` | Initial / peak learning rate for AdamW optimizer | `0.0003` |
| `-T` | `--temp <float>` | Sampling temperature for text generation | `0.8` |
| `-K` | `--topk <int>` | Top-K nucleus filtering for generation | `15` |
| `-h` | `--help` | Display usage summary and exit | |

---

## 🔥 Ready-to-Run Examples

### 1. Multi-Shard Large Dataset Training Workflow (e.g. TinyStories)
Train sequentially across dataset shards (`tinystories/shard_00.txt`, `tinystories/shard_01.txt`, ...) using a pre-defined existing vocabulary file (`--vocab-file`) and continuous Cosine LR schedule (`--total-steps`):

#### Training Shard 00 with a Pre-Defined Vocabulary File (`--vocab-file`)
If you already have a pre-existing vocabulary file (`master_vocab.bin`), pass `--vocab-file` on Shard 00 so `mgpt` loads your existing dictionary instead of rebuilding it:

```powershell
# Shard 00: Trains steps 1 -> 3000 using pre-existing vocabulary file and stable learning rate
.\build_release\Release\mgpt.exe -t -g --vocab-file="master_vocab.bin" -d="tinystories/shard_00.txt" -f="tinystories.bin" --lr=0.0003 -s=3000 --total-steps=72000 -l=12 -c=384 -w=256 -b=16 -a=2
```

*(Note: If `master_vocab.bin` does not exist yet, running Shard 00 without `--vocab-file` will automatically build and save it to `tinystories/shard_00.txt.vocab.bin`).*

#### Resuming on Subsequent Shards (`shard_01.txt` -> `shard_23.txt`)
Use `--resume`, `--vocab-file`, and `--lr` to smoothly continue step counting and Cosine LR decay:

```powershell
# Shard 01: Resumes weights & step count from step 3000 -> 6000
.\build_release\Release\mgpt.exe -t -g --resume -d="tinystories/shard_01.txt" --vocab-file="master_vocab.bin" -f="tinystories.bin" --lr=0.0003 -s=3000 --total-steps=72000 -l=12 -c=384 -w=256 -b=16 -a=2
```

#### Automated Overnight Pipeline across all 24 Shards (Using Pre-Defined Vocab)
```powershell
# Loop through Shards 00 to 23 automatically
0..23 | ForEach-Object {
    $idx = "{0:D2}" -f $_
    $resumeFlag = if ($_ -gt 0) { "--resume" } else { "" }
    Write-Host "=== Training on Shard $idx ===" -ForegroundColor Cyan
    .\build_release\Release\mgpt.exe -t -g $resumeFlag -d="tinystories/shard_${idx}.txt" --vocab-file="master_vocab.bin" -f="tinystories.bin" -s=3000 --total-steps=72000 -l=12 -c=384 -w=256 -b=16 -a=2
}
```

### 2. Supervised Conversational Fine-Tuning (SFT) on Multi-Turn Dialogues
To transform a completed base model checkpoint (`tinystories_shard23.bin`) into a conversational assistant (`mgpt_chat.bin`), fine-tune on a dialogue text file (`dialogues.txt`) formatted with explicit turn markers (`User: ... \n Assistant: ... \n <|endoftext|>`).

Use a low learning rate (`5e-5`) and match the exact architecture parameters (`12L`, `384C`, `256W`, `16B`, `2A`) used during base pretraining:

```powershell
# 1. Copy completed base model weights to a new target checkpoint
Copy-Item -Path "tinystories_shard23.bin" -Destination "mgpt_chat.bin" -Force

# 2. Fine-tune on dialogues.txt for 3,000 steps using --resume with a fresh Cosine LR curve (--start-step=0 --total-steps=3000)
.\build_release\Release\mgpt.exe -t -g --resume -d="dialogues.txt" -f="mgpt_chat.bin" --vocab-file="tinystories_slice.txt.vocab.bin" --lr=0.00005 -s=3000 --start-step=0 --total-steps=3000 -l=12 -c=384 -w=256 -b=16 -a=2

# 3. Test conversational generation with the fine-tuned assistant
.\build\Release\mgpt.exe -i -g -f="mgpt_chat.bin" --vocab-file="tinystories_slice.txt.vocab.bin" -l=12 -c=384 -w=256 -p="User: Hello! How are you today?`nAssistant:" --temp=0.5 --topk=20
```