# MGPT Systems Engineering Chronicle

A chronological log of architectural decisions, performance debugging, and mathematical optimizations for MGPT, a zero-dependency Generative Pre-trained Transformer (GPT) engine built from scratch in C++20 and CUDA C.

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
- **Memory Arena (Scratchpad VRAM Workspace):** Diagnosed severe execution latency caused by `cudaMalloc` and `cudaFree` thrashing inside forward/backward hot paths. Solved this by pre-allocating a set 1 GB VRAM memory arena (`Scratchpad`) during model construction, allowing layers to reuse temporary computation buffers instantly with zero runtime allocation overhead.

### Performance & Results
- **Model Specs:** 6 Layers, 384 Channels, 64 Sequence Length (~10,722,113 parameters).
- **Training Metrics:** Executed 5,000 steps on GPU. Step execution latency dropped to `~144 ms/step` (`800 seconds total duration`), achieving a ~25x throughput acceleration over CPU execution.
- **Training Issues:** Training speed would heavily decline on certain model hyperparameters, which was discovered to be caused by spilling the model on the GPU VRAM to the shared memory.

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
- **Sample Generation:** Transitioned from broken word fragments into coherent, grammatically correct multi-paragraph English stories. Although the model struggled on name generalisation and would default to common names in the dataset.

---

## Phase 5: Algorithmic Arithmetic Specialist & Pipeline Modularization (July 13, 2026)

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
Reduce the VRAM footprint to allow training a 26M parameter model with a 1024 sequence length on consumer hardware, without triggering Out-Of-Memory (OOM) crashes from massive $O(T^2)$ attention matrices.

### Key Engineering Insights
- **Custom Flash Attention Kernels:** Implemented native CUDA `flash_attention_forward` and `flash_attention_backward` kernels, bypassing the allocation of intermediate $T \times T$ attention scores.

### Performance & Results
- **Model Specs:** 16 Layers, 320 Channels (~26,305,942 parameters).
- **VRAM Footprint:** Successfully achieved massive VRAM reductions.
- **Speed Degradation Profile:** 
  - Context Window `256`: `~855ms/step` | `3.7 GB VRAM`
  - Context Window `1024`: `~10,163ms/step` | `8.3 GB VRAM`
- **System Bottleneck:** The current Flash Attention backward pass suffers from extreme global memory contention. Because the grid is parallelized over Q tiles, all blocks fight to update the global `dK` and `dV` matrices using heavy `atomicAdd` calls, scaling at $O(T^2)$ for the backward pass.


## Phase 7: Flash Attention V2 & Unmasking Hidden Bottlenecks (July 25, 2026)

### Goal & Challenge
Eliminate the extreme $O(T^2)$ global memory contention in the backward pass caused by massive `atomicAdd` usage, then verify the speed improvements at `1024` context length.

### Key Engineering Insights
- **FA2 Dual-Kernel Architecture:** Successfully split the single backward pass into two decoupled kernels (`flash_attention_backward_dq_kernel` and `flash_attention_backward_dkv_kernel`). This inverted the loops so each thread block exclusively owns its output tile, completely eliminating `atomicAdd` calls while preserving perfect mathematical parity.
**Scalar Math in Flash Attention (`Fwd: 1478ms`):** The custom Flash Attention kernels (both forward and backward) compute dot products using raw scalar loops (`for (int d=0; d<head_dim; d++)`). Because they don't use blocked cooperative matrix multiplications, the CUDA compiler cannot utilize the RTX 3070's Tensor Cores. The kernels are manually executing 43 Billion math operations on standard CUDA cores, bottlenecking the entire network.


## Phase 8: Legacy Attention VRAM Footprint & Scratchpad Expansion (July 25, 2026)

### Goal & Challenge
Restore the pre-flash-attention legacy looping architecture for the fallback `--no-flash-attention` path, and resolve the severe Scratchpad Memory OOM caused by the massive $O(B \times H \times T^2)$ intermediate tensors used during its backward pass.

### Key Engineering Insights
- **Legacy Fallback Restoration:** Safely restored the loop-based MHA forward and backward passes from Git history (v`f6ef028`) specifically for when Flash Attention is disabled, ensuring architectural parity and giving users a direct cuBLAS scalar execution path.
- **Scratchpad Capacity Ceiling:** Discovered that restoring the legacy backward pass caused immediate `Scratchpad Memory Arena Out of Memory` fatal errors. The legacy path relies on explicit materialization of `cached_probs_T`, `cached_dP`, `cached_dS`, and `cached_dS_scaled_T` matrices on the GPU. For `-b 16` and `-w 1024`, these intermediate tensors combine to demand over `491 MB` of VRAM per layer during backpropagation.
- **Arena Expansion Fix:** Because gradient checkpointing properly frees the Scratchpad at the end of each block, the memory requirement doesn't multiply by the 16 layers. However, the sheer size of a single layer's backward pass required increasing the initial `Trainer` scratchpad capacity from `1 GB (256M floats)` up to `3 GB (768M floats)`.
- **Benchmarking:** This allowed for benchmarking of the old implementation compared to the new flash attention on several context windows and batch sizes, which can be view in $/logs/experiments/$

## Phase 9: Dynamic VRAM Predictor  (July 27, 2026)

### Goal & Challenge
Replace the rigid 3 GB Scratchpad memory arena with a dynamic memory allocator that mathematically predicts exactly how much VRAM the model will need based on its hyperparameters, preventing memory hoarding on small grids and fatal bounds-checking crashes on massive grids.

### Key Engineering Insights
- **Mathematical Peak Prediction:** Implemented a zero-overhead arithmetic formula inside `Trainer.cpp`. This mathematically calculates the peak Scratchpad requirements by summing the matrices (e.g., $16 * B*T*C$ for forward standing stacks).
- **Non-Checkpointing Stack Accumulation:** Discovered that disabling Gradient Checkpointing forced the Scratchpad to hold the forward activations for *all* $L$ layers concurrently. Updated the math predictor to multiply the forward standing block by `config.num_layers` if checkpointing is disabled, which correctly scaled the allocation request from ~70 MB up to ~12 GB.
- **Unified Memory Paging Tolerance:** By properly requesting massive allocations (e.g., 11.9 GB) from the NVIDIA driver rather than failing an internal C++ bounds check, the C++ execution engine successfully delegates overflow to the OS. The driver gracefully spills the excess VRAM over the PCIe bus into System RAM, avoiding an OOM crash entirely (albeit at the cost of significantly increased step times due to PCIe bottlenecking).
- **Virtual Memory Profiler (`--virtual`):** Implemented a `--virtual` flag that executes the Mathematical VRAM Predictor and immediately exits. This allows users to test hyperparameter limits and view exact VRAM requirements without launching the training loop or loading weights.

## Phase 10: Mixture of Experts (August 3, 2026)

### Goal & Challenge
Abstract the MLP layer in the transformer block into a mixture of experts with dynamic routing and load balancing

### Key Engineering Insights
- **Load Balancing** : Some experts were heavily biased and would take all the tokens, requiring a load balancer penalty for the logits
- **Minimise CPU-GPU communication** : Rather than moving the entire tensor to the CPU in order to reorganise the tensors so that the tokens were consecutively grouped by their expert, only a vector<int> of size {experts} was moved from GPU to CPU, in order to calculate a running sum and allocate a tensor that will be used for grouping the original tensor, and provide a map to scatter tokens back to original location
- **Smart Kernel management**: To avoid writing several kernels, we can reuse the gather and scatter kernels in the forward and backwarwd pass and reuse existing kernels like softmax derivative.

## Phase 11: Decoupled Dynamic scratchpad allocation (August 5, 2026)

### Goal & Challenge
Decouple the memory footprint predictor so that Trainer file does not need to change when layers change / are added

### Key Engineering Insights
- **API design** : To hold a data per layer, a struct was used to hold the fwd standing, fwd temp and bwd temp, originally it was fwd peak (which was actually temp + standing) which led to error prone implementations, so a method was introduced in the struct to automatically addt he values and return them

## Phase 12: Profiling (August 6, 2026)

### Goal & Challenge
Using `ncu` and `nsys` to profile the memory allocation bottleneck, and L1/L2 cache throughput and usage to compare the kernel for flash attention on and off and the scratchpad.
**Test Architecture:** NVIDIA RTX 4080 Laptop GPU (CC 8.9), 8 Layers, 320 Channels, 512 Sequence Length, Batch Size 16 (Effective 32). Tested on a 13.1M parameter dense baseline and an 82.1M parameter MoE architecture (8 experts, top-2 routing).

### Key Engineering Insights
- **Flash Attention (`ncu`):** 98.3 KB dynamic shared memory per block. L1 cache throughput increased from 37.28% to 67.08%. L2 cache throughput reduced from 48.11% to 18.51%. Compute SM utilization increased 5.7x (from 2.05% to 11.78%). Total kernel duration: 14.12 ms.
- **Scratchpad (`nsys`):** `cudaMalloc` duration over 10 steps: 369.06 ms. `cudaFree` synchronous WDDM CPU block time avoided: 20.36 seconds per run. Average overhead eliminated: 98.94 ms per step.
- **Gradient Checkpointing (`nsys`):** Peak VRAM reduced by 3.0 GB. Eliminated 2.2 GB of Device-to-Host (PCIe) thrashing that occurs when checkpointing is disabled.
- **MoE Sparsity (`nsys`):** 13.1M parameter dense model (1 expert) executed at ~1.69s/step (372ms Fwd). 82.1M parameter MoE model (8 experts, top-k 2) executed at ~1.83s/step (384ms Fwd). 6.2x parameter scaling at only 8% step latency cost.

## Phase 13: Cleaning (August 10, 2026)

### Goal & Challenge
Clean up the repository, improving old kernels to use tiling and cleaning magic numbers to use config values passed in at runtime

## Todo
- **Floating point memory footprint:** Implement FP16
- **Interactive Chat Mode:** Implement an interactive mode with a Stop Token check for inference.
- **Debug flag** Add debug flag that then prints data to terminal
- **Magic constants** Remove magic constants like channel broadcasting up and down in the self-attention