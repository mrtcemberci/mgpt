# DAY 0 - JULY 5TH 2026

Implemented the tokeniser, works on a character level basis

# DAY 1 - JULY 6TH 2026

Implemented the layers, tensor, the GPT blocks.

Implemented inference that uses the current neural networks forward function and uses the output logits.

Runs only on the CPU, training is incredibly slow.

Below is the initial training output

## Instantiated GPT Model Architecture:
-> Vocab Size:   65
-> Max Seq Len:  64
-> Embed Dim:    128
-> Num Layers:   4
-> Total Params: 818241 float32 parameters (~3196 KB)

Starting Training Loop (AdamW, LR=0.001, Batch=16, Steps=1000)...
------------------------------------------------------------
Step       Progress & Timings                      Train Loss   Val Loss
------------------------------------------------------------
[  1/1000 (  0%)] Fwd:135ms Loss:228ms Opt:26ms | Step:389ms | Loss: 4.9095
[100/1000 ( 10%)] Fwd:115ms Loss:219ms Opt:35ms | Step:370ms | Loss: 2.8100 | Val: 2.7458
[200/1000 ( 20%)] Fwd:116ms Loss:217ms Opt:37ms | Step:372ms | Loss: 2.5868 | Val: 2.5985
[300/1000 ( 30%)] Fwd:119ms Loss:227ms Opt:41ms | Step:388ms | Loss: 2.5108 | Val: 2.5285
[400/1000 ( 40%)] Fwd:117ms Loss:218ms Opt:41ms | Step:377ms | Loss: 2.4345 | Val: 2.4734
[500/1000 ( 50%)] Fwd:117ms Loss:219ms Opt:40ms | Step:378ms | Loss: 2.4079 | Val: 2.4043
[600/1000 ( 60%)] Fwd:118ms Loss:220ms Opt:41ms | Step:379ms | Loss: 2.2644 | Val: 2.3329
[700/1000 ( 70%)] Fwd:116ms Loss:222ms Opt:42ms | Step:381ms | Loss: 2.3066 | Val: 2.2501
[800/1000 ( 80%)] Fwd:120ms Loss:228ms Opt:41ms | Step:390ms | Loss: 2.1144 | Val: 2.2049
[900/1000 ( 90%)] Fwd:116ms Loss:219ms Opt:42ms | Step:378ms | Loss: 2.0759 | Val: 2.1658
[1000/1000 (100%)] Fwd:117ms Loss:222ms Opt:43ms | Step:383ms | Loss: 2.0098 | Val: 2.0490
------------------------------------------------------------
Γ£à Training Complete! Total Duration: 414.55 seconds.

[6/6] Exporting Trained Model & Running Autoregressive Inference Sample...
Successfully saved GPT model weights (70 parameter tensors) to shakespeare_gpt.bin!

--- ≡ƒô£ Text Generation Sample (Prompt: "To be or not to be") ---
To be or not to be my dierobaon the off and that
is for flad cly Men, jyis in theing!
NEMSOMPPETUTQEE:
I ald, no whas I los ble tet you the sWhiller
Have his fraest goders mane poverad,
Sing smin to notior; rem wine Mast, handrees
gke or masion I diand sow! heir hyindow how my's.

DUKUKE VINCETIS:
I, that and it my be hrose! that he deesworjoy!
KING INGD RDERDWARR:
Soboe dids, lend fod on! wa!

CLIZADYRDES:
That way, ast but
That the freemak and the blase died!
Cit, shyall:
But you, batre your with ort.

NOBELAPC

##

Looks sort of like english, this took very long to train because the CPU is slow so I had to lower the GPT specs, that is why it generates garbage.
As you can see, most of the time spent is at the loss step.

I started by refactoring all existing code that makes use of any tensor internals into methods on the tensor class. This makes it easier to port to CUDA, as now I only need to change one file.

**Update**: Completed 100% of the Tensor engine abstraction! Every neural network layer (Linear, GELU, LayerNorm, Embedding, Attention, CrossEntropyLoss) and optimizer (SGD, AdamW) has been refactored to use encapsulated Tensor math methods. All 8 CTest suites pass with 100% success rate.

# TODO:

Implement Tensor on the GPU (CUDA)
Implement a separate inference engine with caching
Implement Multi-Head Attention and RoPE
Build Goku roleplay and Text-to-SQL commercial milestone bots
