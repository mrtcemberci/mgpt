# Architecture of MGPT

This document explains the high-level design decisions, memory management philosophy, and core abstractions used to build this engine.

# Execution

The process can be booted in infer mode or training mode with the appropriate flags (see README) and runs the appropriate pipeline's run
method.

The generator outputs the genarted string with the provided model weights to stdout.

The trainer outputs the model weights into a .bin file with the same file name as the input txt.

# Trainer

Trainer runs regular ML training procedure with cosine-decay learning rate on the model instantiated in the GPT class.

The forward into method for the GPT class takes in a Tensor of BxT and produces a BxTxVocabSize of normalised output for prediction.

The GPT class owns a scratchpad which it passes to its children, this is the memory allocator arena, to determine the size of scratchpad a GPT class needs it recursively asks its children (inner layers) and does the appropriate calculations.

# Layer abstractions

Layers are grouped together in abstractions, for example the transformer block is a a collection of the self-attention and the MLP network, this is used to stack several transformer blocks. GPT is another example of the abstraction.

# Attention

The model (inside transformer block) uses the multi-head attention layer, the multi-head attention layer uses RoPE and a provided number of heads.

# Multi-layer Perceptron

After attention, a Mixture-of-experts layer is used.

# Transformer

The aforementioned parts are joined to form the transformer, which also uses residual skip connections. Root mean squared norms are used
between the connections.

# Optimiser

The optimiser is instanstiated in the trainer file, using a smart pointer, an optimiser has to adhere to the Optimiser abstract class.

# Mathematical Formulation

The engine executes the following formal matrix operations during a single forward pass of the `TransformerBlock`. Let $\mathbf{X} \in \mathbb{R}^{B \times T \times C}$ denote the input tensor.

**1. Pre-Norm & Causal Multi-Head Attention:**
$$ \mathbf{\bar{X}} = \text{RMSNorm}(\mathbf{X}) $$
$$ \mathbf{Q} = \mathbf{\bar{X}}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{\bar{X}}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{\bar{X}}\mathbf{W}_V $$
$$ \mathbf{Q}', \mathbf{K}' = \text{RoPE}(\mathbf{Q}), \text{RoPE}(\mathbf{K}) $$
$$ \text{Attention}(\mathbf{Q}', \mathbf{K}', \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}' (\mathbf{K}')^T}{\sqrt{d_k}} \odot \mathbf{M}\right) \mathbf{V} $$
$$ \mathbf{X}_{att} = \text{Attention}(\mathbf{Q}', \mathbf{K}', \mathbf{V}) \mathbf{W}_O $$
$$ \mathbf{X}_1 = \mathbf{X} + \mathbf{X}_{att} \quad \text{(Residual 1)} $$
*(Note: $\mathbf{M}$ is the lower-triangular causal mask).*

**2. Pre-Norm & Mixture of Experts (SwiGLU):**
$$ \mathbf{\bar{X}}_1 = \text{RMSNorm}(\mathbf{X}_1) $$
For each token $\mathbf{x} \in \mathbf{\bar{X}}_1$, the MoE router selects the top-$k$ experts. For a given expert $E_i$, the SwiGLU activation is computed as:
$$ E_i(\mathbf{x}) = \left( \text{Swish}(\mathbf{x}\mathbf{W}_{1,i}) \odot (\mathbf{x}\mathbf{W}_{V,i}) \right) \mathbf{W}_{2,i} $$
$$ \mathbf{X}_{moe} = \sum_{j=1}^{k} g_j E_j(\mathbf{x}) $$
$$ \mathbf{X}_{out} = \mathbf{X}_1 + \mathbf{X}_{moe} \quad \text{(Residual 2)} $$
