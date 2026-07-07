# 🚀 MGPT: Character-Level GPT Training Engine in C++ & CUDA

A high-performance, zero-allocation character-level Transformer/GPT engine built from scratch in modern C++ and CUDA. Zero Python dependencies, zero external bloat—just pure linear algebra and hardware-accelerated compute kernels.

---

## 🛠️ Building & Compiling (Windows / PowerShell)

### Prerequisites
1. **Microsoft Visual Studio Community 2022** (with *"Desktop development with C++"* workload).
2. **NVIDIA CUDA Toolkit** (v12.x or v13.x installed *after* Visual Studio).
3. **CMake** (included with Visual Studio or standalone).

### Build Command
Open PowerShell in the root directory of the repository and run:

```powershell
# Set your CUDA Toolkit directory (adjust v13.3 to match your installed version, e.g. v12.6)
$env:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\"

# Configure and compile using CMake and MSBuild in Release mode
cmake --build build_msvc --config Release --target mgpt

# Add CUDA binaries to your PowerShell PATH
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64;" + $env:PATH
```

The compiled binary will be located at: `.\build_msvc\Release\mgpt.exe`

---

## 💻 Command-Line Usage

```bash
mgpt [options]
```

### Usage Flags & Configurations

| Flag | Long Flag | Description | Default Value |
| :---: | :--- | :--- | :---: |
| `-t` | `--train` | Run in **Training Mode** (trains model & saves weights) | *Default* |
| `-i` | `--infer` | Run in **Inference Mode** (loads weights & generates text without training) | `false` |
| `-g` | `--gpu` | Enable **CUDA GPU Acceleration** for training and inference | `false` |
| `-f` | `--file <path>` | Path to save/load model weights binary file (`.bin`) | `"shakespeare_gpt.bin"` |
| `-d` | `--data <path>` | Path to training dataset text file | `"input.txt"` |
| `-p` | `--prompt <str>` | Prompt string for autoregressive text generation | `"To be or not to be"` |
| `-n` | `--tokens <int>` | Number of tokens/characters to generate | `500` |
| `-s` | `--steps <int>` | Number of training optimization steps | `1000` |
| `-l` | `--layers <int>` | Number of Transformer blocks ($L$) | `4` |
| `-c` | `--channels <int>` | Embedding channel dimension / hidden size ($C$) | `128` |
| `-b` | `--batch <int>` | Training batch size ($B$) | `16` |
| `-w` | `--window <int>` | Context window / sequence length ($T$) | `64` |
| `-h` | `--help` | Display usage instructions and exit | |

---

## 🔥 Ready-to-Run Commands

### 1. The Full Shakespeare Training Command (Recommended for RTX 3070 / RTX 4080)
Builds and trains a 6-layer, 384-channel GPT model on `input.txt` for 5,000 steps using GPU acceleration, saves the checkpoint to `shakespeare_gpt.bin`, and generates a 1,000-character sample prompt:

```powershell
$env:CudaToolkitDir="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\"; cmake --build build_msvc --config Release --target mgpt; $env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64;" + $env:PATH; .\build_msvc\Release\mgpt.exe -t -g -f="shakespeare_gpt.bin" -s=5000 -l=6 -c=384 -w=256 -b=64 -p="ROMEO:" -n=1000
```

### 2. Generate Text Only (Inference Mode)
Load your trained `shakespeare_gpt.bin` checkpoint and generate dialogue without re-training:

```powershell
.\build_msvc\Release\mgpt.exe -i -g -f="shakespeare_gpt.bin" -p="O Romeo, Romeo! wherefore art thou" -n=1000
```

