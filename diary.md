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

This is the results when ran on GPU with different layers and embed dimensions.

Instantiated GPT Model Architecture:
-> Vocab Size:   65
-> Max Seq Len:  64
-> Embed Dim:    384
-> Num Layers:   6
-> Total Params: 10722113 float32 parameters (~41883 KB)

[4.5/6] Migrating GPT Model and Engine to CUDA GPU...
[5/6] Starting Training Loop (AdamW, LR=0.001, Batch=16, Steps=5000)...
------------------------------------------------------------
Step       Progress & Timings                      Train Loss   Val Loss
------------------------------------------------------------
[  1/5000 (  0%)] Fwd:48ms Loss:105ms Opt:6ms | Step:161ms | Loss: 4.4899
[500/5000 ( 10%)] Fwd:57ms Loss:85ms Opt:0ms | Step:144ms | Loss: 2.5224 | Val: 2.4750
[1000/5000 ( 20%)] Fwd:59ms Loss:90ms Opt:0ms | Step:151ms | Loss: 2.4706 | Val: 2.4193
[1500/5000 ( 30%)] Fwd:58ms Loss:96ms Opt:0ms | Step:156ms | Loss: 2.3663 | Val: 2.3882
[2000/5000 ( 40%)] Fwd:98ms Loss:143ms Opt:0ms | Step:243ms | Loss: 2.3150 | Val: 2.3453
[2500/5000 ( 50%)] Fwd:59ms Loss:87ms Opt:0ms | Step:147ms | Loss: 2.2698 | Val: 2.3091
[3000/5000 ( 60%)] Fwd:74ms Loss:86ms Opt:0ms | Step:162ms | Loss: 2.3076 | Val: 2.2422
[3500/5000 ( 70%)] Fwd:57ms Loss:84ms Opt:0ms | Step:142ms | Loss: 2.2016 | Val: 2.2344
[4000/5000 ( 80%)] Fwd:57ms Loss:86ms Opt:0ms | Step:144ms | Loss: 2.2142 | Val: 2.2382
[4500/5000 ( 90%)] Fwd:58ms Loss:85ms Opt:0ms | Step:144ms | Loss: 2.1809 | Val: 2.1985
[5000/5000 (100%)] Fwd:58ms Loss:85ms Opt:0ms | Step:144ms | Loss: 2.2075 | Val: 2.2133
------------------------------------------------------------
Γ£à Training Complete! Total Duration: 800.47 seconds.

[6/6] Exporting Trained Model to shakespeare_gpt.bin...
Successfully saved GPT model weights (102 parameter tensors) to shakespeare_gpt.bin!

--- ≡ƒô£ Text Generation Sample (Prompt: "ROMEO:") ---
ROMEO:
Dowe facon?

TRI&HARE CTLIEFCWI:
Youo?

PONCUEM:
We on bet; and ringesteer andetry for oum my your
Mesoor nof Cof ungre,
Ris geie, but whe him.

TAJucke.

On Theen nogr in but down cealy', bon'd is, lteave wat.

UShe?

LUSCIULO:F
IITF heXCAn:
Meactins way, our wis
Andy we henos I shous you thap fore a thome,
This pan gareren Is icess, oold sbue wigon hins be lom the fach bith eore

Unaxes oquan, jer aple seyges magrin whe thatche sord upies,
Yemize:
Frrowin:
Shistaing wary! yoweng tri;
Dints ther to thouth sous thate tmar grangss you wiven my en, gray,
But I nothe thy he mor.

BRPRIO:
Thaghthi't to and bier nethat that rous.
Dutry bey roy.

Yous of OF my dowfoNGh
MIO:
qUe gin foull and at do and stuen to prirstes:
Rinstreetst; teaun,
I 'Fr heon,
What his for sund.

Mill to dot it his iw mn, conguevies, word heatit.

ANKthl dohe wakn your wis bre:
Tho sthlow!
VoNut Ang.
DUK:
I this a housh haJone pance,
I way my dat our batel cran buir'ss ainds
Wacks enow she but
are meves.

LUCK:
foll


It still sucks, I believe because the architecture is not advanced enough to hone in on the advanced concepts. 
We are missing Multi-head Attention, and Cosine learning rate decay.

I added both, new problem, training takes way too long, like so long to the point I have not been able to test these new features.
My assumption is the GPU is too fast for the CPU

Suggestion: Get batches in the training loop, i should allocate one giant tensor for this and let it stay on the GPU.
Suggestion: cudaMalloc is thrasing, everytime i do a pass on code it can call cudamalloc, why not preallocate on class construction,
and re use the same tensor. 

# Day 3 : July 7th

I heavily cut down the GPU-CPU transfer during training, but ran into another issue, because I am using 32-bit floating point numbers,
past a certain number of layers and channels, the time to train jumps up significantly. I am pretty sure the issue is regarding the cache.

I need to find a way to scale these, my first goal is byte pair encoding, and to eventually switch from 32 bit to 16 bit floating point numbers.

A simple optimisation I learnt was using 8 bit numbers for adamW values, this did not help at all.


# Day 4 : July 8th

I have made a breakthrough, the issue was not the cache, but actually the VRAM was getting full when increasing the layers/channels, causing the GPU
to use the shared unified memory, which would be much slower to access, dramatically lowering the speed of training, I monitored task manager
during training and surely this was the case, and halving the batch size immediately brought training back to great speeds.

I have researched fixes, which include trimming down temporary tensors, gradient accumulation and gradient checkpointing.

I implemented gradient accumulation, and a scratch memory VRAM workplace, for layers to re-use temporarily for calculations, this cut down the memory footprint
while also not requiring cuda allocations during hot paths. I ran into an issue where I had to add the ability to pass in a float* to view as a tensor
this led to an issue with the CPU as it used vectors rather than direct addresses, hence the tensor::view function is limited to cuda devices only.

This cut down the memory footprint by almost 3GB!!!!!!!

In addition, I implemented Block-Level Gradient Checkpointing (Activation Checkpointing). Instead of retaining all forward intermediate activations across all Transformer blocks simultaneously (~1.8GB for L=6), each block now stores only its input tensor X_l. During backpropagation, each block recomputes its internal activations just-in-time from its input before computing parameter gradients.

# Day 5 : July 9th

I implemented BPE tokenizing, I ran the model with 512 vocab size, 16 batch size (effective x2), 1200 steps, 384 channels, 256 token context window, and these were the results:

ROMEO: your graland;
Aning my kill mfoes toge'such as toged, I, yet bel;
Huns, sil, Sin me garets to defer lack here
To hear yourself with them ondishon. We have becomell have ake?

ISHERBELL:
Lly without like a cheldful ga tal seed kingly stand,
When say they rengractors an overs;
rave I have prisoner hope, I send me deign of the world,
Even in so more so aught their fets with wraines,
I think mistall ract an not the vedyself,
ungetison I prace be night.

LEONTES:
Who same have beseed agmince within
I am loved with such formsue
Thairly not abattual soted na thoufffore,
When thou meet which am countince
I here trif of th thou wbe sfor the cause yons
As if you myself beliew not exceive me the doess,
But for tearsing be his mon'd of hand,
Hearer thee:
What did the Buse of the gods,
Fromoes in magood posceive,
Cristild out his founsels and feast,
'Tis my made you should g one; or on men no
I should marreat his just conceed,
To would would seembaroin at exe,
And not stay all the world,
Romin the meet and hand?
I thee be thy counc went so next too mate,
Have while,
A tope but have set, on his name, keal jest.

LORTIUS:
A good cales--

AUFII:
Ay, comes to Fnow:
I shall you hast so. Cry not
Cell me your wintes!
God the ear, in Rome! you fl! where is thoua luice,
Of you, here is a lip's feed?

LTA:
Alaster? Fare to your gue,
A soltly husband.

LEONTES:
I have bety t's day?

ANTES:
What, be but den.

ANTES:
You'll fear say,
I make this us a offereld: dce with a truitor
Greach as joio the truwearst: though Thir art y.

y hold
The noble that hence come to may, She had you, I know
The firction is your helder that.

HASTER:
Mmabrain theem to your hop are you woff
That his consequal?

ASONGELY:
Graiance, to estir, my you INCENTIO:
Your grace, will pracuse you be infits,
A supmen oneyou, if you know you was gright.

HASTINGS:

Pray, I waldister-cor foo'd more, move.
Nowingult you a worthy comppersons? and well.
Let you all a foolihfauly, shie, in jestee
Morrent be int. Parry, Looem knows habo'
Ra calone and cank hair in the preport nack's mondery dear;
Are for hour shaking's ancius: y' before my brother,
Prain me requick,'st en toio-body.

PAULINA:
Go! A should lieve at your ba hon Brunt,
I come my scarted but to
------------------------------------------------------------

With T=0.7 and K=10

KING RICHARD: Speak youKill; ford for a king
To lovEyour own paces,
Bardintune will requit forbout of York
To before joy; for her Camillo can;
Shall eyes him disder my land?'ll me, Clarences
To depose: brother we that stigpose
Pece needutince he plains out that he hath affinxess,
One an and, 'erd put ity, halfly,
Be interinghaught time, Still. Romey, diss?
That's heart, hury, but is othere.

FRIAR OVOLURes OMhat's hblant:
What sure, mother, a reservanting of wall.

Shaor:
Signi' hars ags, and I cheek's ppears,
Tis as thou in simateld; and eedly disp,
Thur? Bring soneseech bege now, yoners
Biring heam bl eart: bless Peach toable quain:
Are is this micked by monINGwinisted
So did toe thee abin a night.
Well, but aither? we me, they was, usin the with
With that scuty the mastere with his face,
And my fellow cound wans and a can pered hap misin.

CAMus:
No, thou crin the heart hastands of boody fewell be sight.
Tell, brothlen ifevenupon me dayINIUS:
Withald, we word tears
From them, and degive me to be to thee:
So wear the whole, his own chee
And to thy lord to aold a vire: warry reams
I am shapatty pare and power, ourze of the minct
This good cound purpose.

Second Servingman:
'Tis not came as he shall ken wors,
Which hadomarET doth begUCKINGHovery never appear thee-may to depreckemses,
That bnilds your here and attread?

Putior XEN:
Marey, as the swears oftle.

First MAusicient lone spo;
To brince too a grave melemy lords; years and fair me
To my law such pery haong art at here!
Bear Good to the sonle!

AAUFirst Servant:
I will? one anion here he migh in hears of the staring
Gake, oo, found and mewere ive.

MAll:
Why, 'tis tell been proted we be formemiof a vison:
The whiest of my laughtered ORI,
The stays the tootarto righ wall ite
Which he is plexfor this gosound less are a
------------------------------------------------------------

# TODO:

Implement Tensor on the GPU (CUDA)
Implement a separate inference engine with caching
Implement Multi-Head Attention and RoPE
Build Goku roleplay and Text-to-SQL commercial milestone bots
