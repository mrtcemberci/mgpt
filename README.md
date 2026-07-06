## Command-Line Usage

```bash
mgpt [options]
```

### Usage Flags

| Flag | Long Flag | Description | Default Value |
| :---: | :--- | :--- | :---: |
| `-t` | `--train` | Run in **Training Mode** (trains model & saves weights) | *Default* |
| `-i` | `--infer` | Run in **Inference Mode** (loads weights & generates text) | `false` |
| `-f` | `--file <path>` | Path to model weights binary file (`.bin`) | `"shakespeare_gpt.bin"` |
| `-d` | `--data <path>` | Path to training dataset text file | `"input.txt"` |
| `-p` | `--prompt <str>` | Prompt string for autoregressive text generation | `"To be or not to be"` |
| `-n` | `--tokens <int>` | Number of tokens/characters to generate | `500` |
| `-s` | `--steps <int>` | Number of training optimization steps | `1000` |
| `-l` | `--layers <int>` | Number of Transformer blocks ($L$) | `4` |
| `-c` | `--channels <int>` | Embedding channel dimension ($C$) | `128` |
| `-h` | `--help` | Display usage instructions and exit | |

---

### Examples

#### 1. Train a New Model
Train a 4-layer model for 5,000 steps on `input.txt` and save weights to `my_model.bin`:
```bash
mgpt -t -f="my_model.bin" -s=5000 -l=4 -c=128
```

#### 2. Generate Text (Inference Only)
Load a previously trained model from `my_model.bin` without retraining, and generate 1,000 characters from a custom prompt:
```bash
mgpt -i -f="my_model.bin" -p="O Romeo, Romeo! wherefore art thou" -n=1000
```
