# MGPT Systems Engineering Chronicle

A chronological log of architectural decisions, performance debugging, and mathematical optimizations for MGPT—a zero-dependency Generative Pre-trained Transformer (GPT) engine built from scratch in C++20 and CUDA C.

---

## Phase 1: CPU Genesis & Character Tokenization (July 5 - July 6, 2026)

### Goal & Challenge
Build a complete, standalone GPT architecture (forward pass, backward autograd engine, and optimization step) from scratch in C++ without external deep learning frameworks.

### Key Engineering Insights
- **Polymorphic Layer Architecture:** Designed `Layer` abstract base class defining `forward()`, `backward()`, `get_params()`, and `get_grads()` contracts to unify all neural network transformations (`LinearLayer`, `EmbeddingLayer`, `GELULayer`, `RMSNormLayer`, `MultiHeadAttentionLayer`).
- **Character Tokenization:** Implemented a direct character-level mapping (`CharacterTokenizer`) to process raw ASCII text strings into integer token IDs and reconstruct text from output probabilities.

### Performance & Results
- **Model Specs:** 4 Layers, 128 Channels, 64 Sequence Length (~3.2 MB / 818,241 parameters).
- **Training Metrics:** Executed 1,000 steps on `input.txt` (Shakespeare corpus). Validation loss converged from `4.90` down to `2.04`.
- **System Bottleneck:** Single-threaded CPU execution averaged `~380 ms/step` for a miniature model, proving that scaling sequence length and channel depth would require hardware parallelization.

---

## Phase 2: CUDA GPU Migration & Memory Arena (July 7, 2026)

### Goal & Challenge
Port all core mathematical operations and tensor manipulations from CPU host memory to NVIDIA CUDA GPU execution (`RTX 3070`) while scaling the architecture to ~10.7 million parameters.

### Key Engineering Insights
- **Dual-Device Tensor Container:** Refactored `Tensor` class to manage both host (`float* data`) and device pointers, incorporating `to_device()` and `to_host()` methods for clean PCIe memory migration.
- **Custom CUDA Kernels:** Wrote native CUDA C kernels (`cuda_matmul`, `cuda_gelu`, `cuda_rmsnorm`, `cuda_softmax`) to execute matrix operations using shared-memory tiling and warp-level parallel threads.
- **Memory Arena (Scratchpad VRAM Workspace):** Diagnosed severe execution latency caused by `cudaMalloc` and `cudaFree` thrashing inside forward/backward hot paths. Solved this by pre-allocating a 1 GB VRAM memory arena (`Scratchpad`) during model construction, allowing layers to reuse temporary computation buffers instantly with zero runtime allocation overhead.

### Performance & Results
- **Model Specs:** 6 Layers, 384 Channels, 64 Sequence Length (~10,722,113 parameters).
- **Training Metrics:** Executed 5,000 steps on GPU. Step execution latency dropped to `~144 ms/step` (`800 seconds total duration`), achieving a ~25x throughput acceleration over CPU execution.

---

## Phase 3: VRAM Unified Memory Spillover & Block-Level Gradient Checkpointing (July 8, 2026)

### Goal & Challenge
Scale model depth and sequence context window without exceeding the physical 8 GB VRAM capacity of the consumer NVIDIA RTX 3070 GPU.

### Key Engineering Insights
- **Diagnosing Memory Contention:** Increasing model layers and sequence lengths caused step execution latency to suddenly spike by over 500%. Hardware monitoring revealed that when 8 GB VRAM capacity was exceeded, the CUDA driver silently paged intermediate activation tensors across the PCIe bus into host system RAM (Unified Memory thrashing), devastating memory bandwidth.
- **Block-Level Gradient Checkpointing (Activation Recomputation):** Implemented custom memory checkpointing across the `TransformerBlock` stack. During the forward pass, interior layer activations (`Attention`, `MLP`) are discarded immediately after computation; only the input boundary tensor `X_l` of each block is retained in VRAM. During the backward pass, interior activations are dynamically recomputed just-in-time directly from `X_l` before computing parameter gradients.

### Performance & Results
- **Memory Footprint Reduction:** Slashed VRAM memory consumption during backpropagation by **~3.0 GB**, eliminating Unified Memory paging entirely and restoring full CUDA core occupancy at double the context window size.
- **Gradient Accumulation:** Added micro-batch gradient accumulation (`-a` flag) to decouple physical GPU memory limits from effective mathematical batch sizes (`Batch 16 x Accum 2 = Effective Batch 32`).

---

## Phase 4: BPE Tokenization, Relative Positional Embeddings & Sharded Training (July 9, 2026)

### Goal & Challenge
Upgrade tokenization efficiency and positional awareness to train on larger, multi-shard corpora (`TinyStories` dataset) without word-fragmentation or sequence drift.

### Key Engineering Insights
- **Regex-Protected Byte Pair Encoding (`BytePairEncodingTokenizer`):** Replaced character-level mapping with subword BPE tokenization (`4,096 vocab size`), utilizing regular expression boundaries (`\r\n`, `[A-Za-z]+`, `\d+`) to prevent unnatural character merging across punctuation and whitespace.
- **Relative Positional Awareness (`RoPE`-Style):** Transitioned from absolute lookup tables to relative sequence positioning within the attention calculation (`MultiHeadAttentionLayer`), improving generalization across variable sequence lengths.
- **Binary Stream Sharding:** Developed raw binary token caches (`*.tok.bin`) to dump pre-encoded integer arrays directly to disk, reducing startup dataset initialization times from minutes down to milliseconds across multi-gigabyte training splits.

### Performance & Results
- **Model Specs:** 12 Layers, 384 Channels, 256 Sequence Length (~31,531,264 parameters).
- **Training Metrics:** Trained continuously across 7 incremental dataset shards (`72,000 total steps`). Validation loss converged from initial `3.54` down to `1.48` on `shard_02`.
- **Sample Generation:** Transitioned from broken word fragments into coherent, grammatically correct multi-paragraph English stories.

---

## Phase 5: Algorithmic Arithmetic Specialist & Pipeline Modularization (Recent)

### Goal & Challenge
Train the Transformer architecture to perform precise, multi-digit algorithmic reasoning (`+`, `-`, `*`, `/`) via Chain-of-Thought scratchpad generation without digit hallucination, and refactor the monolithic application controller (`main.cpp`).

### Key Engineering Insights
- **1-Character Structural Tokenization:** Designed explicit single-character delimiters (`?` for query, `{` for scratchpad start, `}` for scratchpad close, `!` for step indicator, `;` for end of sequence) to guarantee 100% inference compatibility with `BytePairEncodingTokenizer::encode` without stripping symbols or changing C++ parsing logic.
- **Multi-Scale Variable-Length Scratchpads:** Generated 500,000 multi-scale arithmetic problems (1 to 4 digits) featuring explicit partial product summation lines (`{...}`) inside the context window. This trains self-attention to treat the scratchpad as a variable-length state machine rather than memorizing fixed positional indices.
- **Context Window Optimization (`-w=128`):** Analysis of token distribution verified that the longest 3-digit multiplication problem requires only 94 tokens. Reducing the context window from 256 to 128 accelerated self-attention matrix operations (`O(N^2)`) by 4x while maintaining a 34-token safety buffer.
- **Pipeline Modularization:** Split the 638-line `main.cpp` controller into clean, modular components (`CLIConfig.h`, `Trainer.cpp/.h`, and `Generator.cpp/.h`) to isolate dataset batching, validation evaluation, and autoregressive Top-K/Temperature sampling into dedicated execution pipelines.

### Performance & Results
- **Algorithmic Accuracy:** Confirmed exact mathematical convergence and multi-digit carry propagation on out-of-distribution test problems (`log_three.txt`).
- **Hardware Efficiency:** Achieved continuous `~99% CUDA GPU occupancy` at `~212 ms/step` (`Batch 16, Accum 2`) on an NVIDIA RTX 3070.


## Phase 6: Custom Flash Attention & VRAM Optimization (July 24, 2026)

### Goal & Challenge
Drastically reduce the VRAM footprint to allow training a 26M parameter model with a 1024 sequence length on consumer hardware, without triggering Out-Of-Memory (OOM) crashes from massive $O(T^2)$ attention matrices.

### Key Engineering Insights
- **Custom Flash Attention Kernels:** Implemented native CUDA `flash_attention_forward` and `flash_attention_backward` kernels, bypassing the allocation of intermediate $T \times T$ attention scores.
- **Scratchpad Lifecycle Fix:** Discovered a severe memory leak during the validation evaluation loop. The `Scratchpad` memory arena was steadily growing because the `savepoint` was never restored during validation inference. Fixed by injecting a targeted `model.reset_activations()` sweep at the end of the evaluation loop, locking the VRAM footprint completely.

### Performance & Results
- **Model Specs:** 16 Layers, 320 Channels (~26,305,942 parameters).
- **VRAM Footprint:** Successfully achieved massive VRAM reductions.
- **Speed Degradation Profile:** 
  - Context Window `256`: `~855ms/step` | `3.7 GB VRAM`
  - Context Window `1024`: `~10,163ms/step` | `8.3 GB VRAM`
- **System Bottleneck:** The current Flash Attention backward pass suffers from extreme global memory contention. Because the grid is parallelized over Q tiles, all blocks fight to update the global `dK` and `dV` matrices using heavy `atomicAdd` calls, scaling terribly at $O(T^2)$.

### Goal Fix: Flash Attention V2
- **Next Step:** Implement a Flash Attention V2-style dual-kernel backward pass. By splitting the backward pass into two separate kernels (one looping over K to compute `dQ` without atomics, and one looping over Q to compute `dK/dV` without atomics), we will completely eliminate global memory contention and restore sub-second step times for large sequence lengths.

## Phase 7: Flash Attention V2 & Unmasking Hidden Bottlenecks (July 25, 2026)

### Goal & Challenge
Eliminate the extreme $O(T^2)$ global memory contention in the backward pass caused by massive `atomicAdd` usage, then verify the speed improvements at `1024` context length.

### Key Engineering Insights
- **FA2 Dual-Kernel Architecture:** Successfully split the single backward pass into two decoupled kernels (`flash_attention_backward_dq_kernel` and `flash_attention_backward_dkv_kernel`). This inverted the loops so each thread block exclusively owns its output tile, completely eliminating `atomicAdd` calls while preserving perfect mathematical parity.
- **Unmasking Hidden Bottlenecks:** While the backward pass (`Loss`) dropped from `1986ms` to `1396ms` (a 30% reduction), the overall Step time remained shockingly high at `8295ms`. This revealed two massive architectural flaws that were previously hidden by the `atomicAdd` stall:
  1. **Optimizer Sync Stall (`Opt: 2137ms`):** Gradient clipping in the optimizer uses `cuda_ops::sum_squares` to sum the parameter gradients. This function calls `cublasSdot` with a CPU host pointer `&result`. With 197 parameters in the model, this forces the CPU and GPU to hard-synchronize 197 times every single step, destroying parallel execution and artificially bloating the optimizer step to over 2 seconds.
  2. **Scalar Math in Flash Attention (`Fwd: 1478ms`):** Our custom Flash Attention kernels (both forward and backward) compute dot products using raw scalar loops (`for (int d=0; d<head_dim; d++)`). Because they don't use blocked cooperative matrix multiplications, the CUDA compiler cannot utilize the RTX 3070's Tensor Cores. The kernels are manually executing 43 Billion math operations on standard CUDA cores, bottlenecking the entire network.

### Goal Fix: Optimizer Device-Side Reduction
- **Next Step:** Implement a custom CUDA kernel to compute the sum of squares entirely on the device without halting the CPU. This will eliminate the 197 host syncs and instantly drop the optimizer step from `2.1 seconds` down to a few milliseconds.

## Phase 8: Legacy Attention VRAM Footprint & Scratchpad Expansion (July 25, 2026)

### Goal & Challenge
Restore the pre-flash-attention legacy looping architecture for the fallback `--no-flash-attention` path, and resolve the severe Scratchpad Memory OOM caused by the massive $O(B \times H \times T^2)$ intermediate tensors used during its backward pass.

### Key Engineering Insights
- **Legacy Fallback Restoration:** Safely restored the loop-based MHA forward and backward passes from Git history (v`f6ef028`) specifically for when Flash Attention is disabled, ensuring architectural parity and giving users a direct cuBLAS scalar execution path.
- **Scratchpad Capacity Ceiling:** Discovered that restoring the legacy backward pass caused immediate `Scratchpad Memory Arena Out of Memory` fatal errors. The legacy path relies on explicit materialization of `cached_probs_T`, `cached_dP`, `cached_dS`, and `cached_dS_scaled_T` matrices on the GPU. For `-b 16` and `-w 1024`, these intermediate tensors combine to demand over `491 MB` of VRAM per layer during backpropagation.
- **Arena Expansion Fix:** Because gradient checkpointing properly frees the Scratchpad at the end of each block, the memory requirement doesn't multiply by the 16 layers. However, the sheer size of a single layer's backward pass required increasing the initial `Trainer` scratchpad capacity from `1 GB (256M floats)` up to `3 GB (768M floats)`.

### Goal Fix: Optimizer Device-Side Reduction
- **Next Step:** Return to the hidden performance bottlenecks uncovered in Phase 7 and implement the custom CUDA kernel to compute the sum of squares entirely on the device without halting the CPU.

## Todo
- **Floating point memory footprint:** Implement FP16
- **Dynamic scratch pad allocation:** Implement dynamic scratch pad allocation rather than a set 1GB
