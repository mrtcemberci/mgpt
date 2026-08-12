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


