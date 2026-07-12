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

The model has gone from spitting garbage, to mostly real text by updating the BPE encoding to prevent fusing of incorrect words by using REGEXs.

--- ≡ƒô£ Text Generation Sample (Prompt: "KING RICHARD: Speak" | Temp: 0.7 | Top-K: 15) ---
KING RICHARD: Speak my daughter's banishment;
Here have, the little, i' the wife,
For I to keep my part, I'll perhalf,
But in the chair of this brailt,
I'll labbour my humy of your highness.

RICHARD:
WARWas each, no more words:
I would not stand for adver'd with woe:
And, my lord, and I shall be smildly,
Shall boy, by my imprembles,
Whose friendly prepare my wronger'd up the find:
I darell, by my signity, for I,
That thought on thy life.

RICHARD:
I will not dost, and damned, and I will.
And, Lord Marius, and I am Hereford,
Not before, and menenianus and Duke of Gloucester,
Or Duke of Gloucester,
I cannot law the first, but I may breaks
In this impery.

YORK:
Alas, my lord, my lord, you have done,
With scorned to be closed.

GLOUCESTER:
Who cousin? what is this?

GLOUCESTER:
Now, better Clift, and I know it.

LADY ANNE:
It is a goodness, good Perder'd.

GLOUCESTER:
I shall tell thee that we were please,
I have not so. I will be put you to be a mercy.

GLOUCESTER:
I would say you, you must do.

GLOUCESTER:
What say, I'll be gone.

YORK:
It is a man, my grace of my mother;
Here company, the ta'en singled instrument,
I am reclaimed in the state,
And plead in this retorted to bed the duke.

GLOUCESTER:
How do you not a words:
I know not all deserved, and I will storm'd.

GLOUCESTER:
I know you do.

GLOUCESTER:
I will, no word Angelo.

YORK:
I have you will please of King of this,
Hower, that which is my lord's cousin.

GLOUCESTER:
YORKing Henry, and my lord,
Go, to have pardon meet to do them, the king.

YORK:
How cannot do you do to me?

GLOUCESTER:
I am so, for I know not speak to do it.

WARWICK:
I will not pute for a child my word:
And if you sorrow to the king;

I removed the absolute position embedding table and switched to a RoPE method for positioning, and the validation loss was around 3.4, with the following text produced
T=0.8 K=15

ROMEO:
I think I say, that's the body and leave.

LADY GREY:
I have been so, and I may, my grace my
signier's baw to-derving to him?

SICINIUS:
By't,
Not a watery and leave the troose.

MENENIUS:
They thrived to see him a premit you:
Sthat I am a crambersy,
To purn him to the duke in actions,
Shap the warrant of his water'd in the patient,
His carried sleeping captain of his house,
She's appear and lady, and I fear
And, whom air, and wife, and framness
And better the law shall perfall you.

Second Servingman:
Is the king is nothing?

BRUTUS:
Sirrah, I am the people, and close of call,
Which you shall not fear the witness of came,
And could not on them for this fearful mind,
To make the pupose to begy to perfection,
And to begguediply to beged me:
For his remeds, that, the bosom of Tybalt,
Against my very part, indeed to her tried?

KING RICHARD III:
I'll be contrrefite my heart.

KING RICHARD III:
I'll not my wife.

MENENIUS:
Go he not, to your water:
But, and dulls I will pardon.

BRUTUS:
One, my lord, I may say you have not.

KING RICHARD III:
Ay, I am a cased and perceive you,
Then called for the people, the leann
Of your gracious matter, I am relieve you are.

RICHMOND:
Ay, I am call the Juliety of the fearful hand,
Sacy to-morrow, I'll be a wronger,
Her mother the devil the greateral tears,
Or I will diedy, and to pail me,
Who souch'd with you.

KING RICHARD III:
Now, as it.

CAMILLO:
Yea,
Is not the France of you?

MIANLEY:
Have you, that I have young the duke of me
Well, siring, as I cannot being
And banch you.

BUCKINGHAM:
What's head?

POLIXENES:
Beformenty!

MENENIUS:
It is my mother,
I have bitch'd the senter of my wife,
I think it is noble mates: though'st thou
Twould you not the sound, and, and I have keeps
To two the matter into my county,
Being in the chamberal'd to case;
And, and first weep tears and watering coldily!

MENENIUS:
I write, my lord.

DUKE VINCENTIO:
The sley of it!

BAPTISTA:
My lord, before, I'll change, sir; my wife?

JULIET:
No, thou hast for thy name!

-----------------------------------------

Training on the tiny stories snippet of 20MB with 4000 steps produced 2.35 ish loss and the following outputs.

Once upon a time, the boy smelt really wet. He had a big smile on his face. He liked the view of the coral.
At the end of the day, the boy was so tired that he was happy. He wanted to go back to the park. So, he ran as fast as he could. But no matter how hard she tried, he was so strong.

The blonde girl and the rabbit. Every day, the rabbit would go to sleep and smile.
One day, the bunny was feeling tired. He went to the forest to rest. He saw a big tree and he thought it would be fun to play on the tree. So the bunny hopped back to his tree and gave him a carrot.
The bunny was very happy and gave the bunny a warm blanket. Then he went to have a picnic and it was so happy. He had a great day!

--------------------------

Migrating GPT Model and Engine to CUDA GPU...
Resuming Training Mode: Loading existing checkpoint from tinystories.bin...
Successfully loaded GPT model weights from tinystories.bin (restored at completed step 3000)!
Uploading Train and Validation Datasets to GPU Memory...
Starting Training Loop (AdamW, LR=0.001, Micro-Batch=16, Accum Steps=2 [Effective Batch: 32], Shard Steps=3000, Global Horizon=3000 -> 6000/72000)...
------------------------------------------------------------
Step       Progress & Timings                      Train Loss   Val Loss
------------------------------------------------------------
[3001/72000 (Shard 1/3000 |   4%)] LR:8.34e-04 Fwd:187ms Loss:137ms Opt:8ms | Step:506ms | Loss: 2.3385
[3300/72000 (Shard 300/3000 |   4%)] LR:9.17e-04 Fwd:3ms Loss:160ms Opt:0ms | Step:539ms | Loss: 2.3270 | Val: 2.1570
[3600/72000 (Shard 600/3000 |   5%)] LR:1.00e-03 Fwd:3ms Loss:159ms Opt:0ms | Step:540ms | Loss: 2.0127 | Val: 2.0779
[3900/72000 (Shard 900/3000 |   5%)] LR:1.00e-03 Fwd:3ms Loss:159ms Opt:0ms | Step:538ms | Loss: 2.0747 | Val: 2.0078
[4200/72000 (Shard 1200/3000 |   5%)] LR:1.00e-03 Fwd:3ms Loss:158ms Opt:0ms | Step:543ms | Loss: 1.9652 | Val: 1.9979
[4500/72000 (Shard 1500/3000 |   6%)] LR:1.00e-03 Fwd:3ms Loss:158ms Opt:0ms | Step:539ms | Loss: 1.8309 | Val: 1.9426
[4800/72000 (Shard 1800/3000 |   6%)] LR:9.99e-04 Fwd:3ms Loss:158ms Opt:0ms | Step:541ms | Loss: 1.9969 | Val: 1.8726
[5100/72000 (Shard 2100/3000 |   7%)] LR:9.99e-04 Fwd:3ms Loss:158ms Opt:0ms | Step:538ms | Loss: 1.8542 | Val: 1.8378
[5400/72000 (Shard 2400/3000 |   7%)] LR:9.98e-04 Fwd:3ms Loss:159ms Opt:0ms | Step:541ms | Loss: 1.8363 | Val: 1.8782
[5700/72000 (Shard 2700/3000 |   7%)] LR:9.98e-04 Fwd:3ms Loss:159ms Opt:0ms | Step:539ms | Loss: 1.7907 | Val: 1.8464
[6000/72000 (Shard 3000/3000 |   8%)] LR:9.97e-04 Fwd:3ms Loss:159ms Opt:0ms | Step:540ms | Loss: 1.8937 | Val: 1.9429

Migrating GPT Model and Engine to CUDA GPU...
Resuming Training Mode: Loading existing checkpoint from tinystories.bin...
Successfully loaded GPT model weights from tinystories.bin (restored at completed step 6000)!
Uploading Train and Validation Datasets to GPU Memory...
Starting Training Loop (AdamW, LR=0.001, Micro-Batch=16, Accum Steps=2 [Effective Batch: 32], Shard Steps=3000, Global Horizon=6000 -> 9000/72000)...
------------------------------------------------------------
Step       Progress & Timings                      Train Loss   Val Loss
------------------------------------------------------------
[6001/72000 (Shard 1/3000 |   8%)] LR:9.97e-04 Fwd:191ms Loss:118ms Opt:8ms | Step:492ms | Loss: 1.9741
[6300/72000 (Shard 300/3000 |   8%)] LR:9.97e-04 Fwd:3ms Loss:158ms Opt:0ms | Step:538ms | Loss: 1.9614 | Val: 1.8600
[6600/72000 (Shard 600/3000 |   9%)] LR:9.96e-04 Fwd:3ms Loss:158ms Opt:0ms | Step:540ms | Loss: 2.1010 | Val: 2.1048
[6900/72000 (Shard 900/3000 |   9%)] LR:9.95e-04 Fwd:3ms Loss:159ms Opt:0ms | Step:538ms | Loss: 2.2186 | Val: 2.1925
[7200/72000 (Shard 1200/3000 |  10%)] LR:9.94e-04 Fwd:3ms Loss:161ms Opt:0ms | Step:542ms | Loss: 2.2690 | Val: 2.4271
[7500/72000 (Shard 1500/3000 |  10%)] LR:9.93e-04 Fwd:3ms Loss:158ms Opt:0ms | Step:538ms | Loss: 2.3863 | Val: 2.4103
[7800/72000 (Shard 1800/3000 |  10%)] LR:9.92e-04 Fwd:3ms Loss:158ms Opt:0ms | Step:540ms | Loss: 2.4259 | Val: 2.5911
[8100/72000 (Shard 2100/3000 |  11%)] LR:9.90e-04 Fwd:3ms Loss:160ms Opt:0ms | Step:538ms | Loss: 3.2283 | Val: 3.1923
[8400/72000 (Shard 2400/3000 |  11%)] LR:9.89e-04 Fwd:3ms Loss:159ms Opt:0ms | Step:539ms | Loss: 3.0228 | Val: 3.0936
[8700/72000 (Shard 2700/3000 |  12%)] LR:9.88e-04 Fwd:3ms Loss:159ms Opt:0ms | Step:542ms | Loss: 3.6631 | Val: 3.6165
[9000/72000 (Shard 3000/3000 |  12%)] LR:9.86e-04 Fwd:3ms Loss:159ms Opt:0ms | Step:539ms | Loss: 3.4204 | Val: 3.5227

igrating GPT Model and Engine to CUDA GPU...
Resuming Training Mode: Loading existing checkpoint from tinystories.bin...
Successfully loaded GPT model weights from tinystories.bin (restored at completed step 9000)!
Uploading Train and Validation Datasets to GPU Memory...
Starting Training Loop (AdamW, LR=0.001, Micro-Batch=16, Accum Steps=2 [Effective Batch: 32], Shard Steps=3000, Global Horizon=9000 -> 12000/72000)...
------------------------------------------------------------
Step       Progress & Timings                      Train Loss   Val Loss
------------------------------------------------------------
[9001/72000 (Shard 1/3000 |  12%)] LR:9.86e-04 Fwd:193ms Loss:121ms Opt:8ms | Step:501ms | Loss: 3.5416
[9300/72000 (Shard 300/3000 |  12%)] LR:9.85e-04 Fwd:3ms Loss:158ms Opt:0ms | Step:541ms | Loss: 3.5292 | Val: 3.5369
[9600/72000 (Shard 600/3000 |  13%)] LR:9.83e-04 Fwd:3ms Loss:160ms Opt:0ms | Step:539ms | Loss: 3.1079 | Val: 3.2108
[9900/72000 (Shard 900/3000 |  13%)] LR:9.81e-04 Fwd:3ms Loss:159ms Opt:0ms | Step:542ms | Loss: 3.2823 | Val: 3.2442
[10200/72000 (Shard 1200/3000 |  14%)] LR:9.79e-04 Fwd:3ms Loss:158ms Opt:0ms | Step:539ms | Loss: 3.3543 | Val: 3.4565
[10500/72000 (Shard 1500/3000 |  14%)] LR:9.78e-04 Fwd:3ms Loss:159ms Opt:0ms | Step:542ms | Loss: nan

Using dataset at: C:\Users\bogaz\MGPT\mgpt\tinystories\shard_06.txt
Found cached binary dataset files:
      -> Loaded Vocabulary Size = 4096 tokens (tinystories_slice.txt.vocab.bin)
      -> Loading pre-encoded token stream from C:\Users\bogaz\MGPT\mgpt\tinystories\shard_06.txt.tok.bin...
      Total Dataset Tokens: 24130492 (Loaded instantly from binary cache!)
Train/Val Split (90/10):
      -> Training Tokens:   21717442
      -> Validation Tokens: 2413050

Instantiated GPT Model Architecture:
      -> Vocab Size:    4096
      -> Max Seq Len:   256
      -> Embed Dim:     384
      -> Num Layers:    12
      -> Checkpointing: ENABLED (Block-Level Activation Recomputation)
      -> Total Params:  31531264 float32 parameters (~123169 KB)

Migrating GPT Model and Engine to CUDA GPU...
Resuming Training Mode: Loading existing checkpoint from tinystories_shard06.bin...
Successfully loaded GPT model weights from tinystories_shard06.bin (restored at completed step 15000)!
Uploading Train and Validation Datasets to GPU Memory...
Successfully loaded AdamW optimizer state (step 3000) from tinystories_shard06.bin.opt!
Starting Training Loop (AdamW, LR=0.0003, Micro-Batch=16, Accum Steps=2 [Effective Batch: 32], Shard Steps=3000, Global Horizon=15000 -> 18000/72000)...
------------------------------------------------------------
Step       Progress & Timings                      Train Loss   Val Loss
------------------------------------------------------------
[15001/72000 (Shard 1/3000 |  20%)] LR:2.82e-04 Fwd:187ms Loss:121ms Opt:194ms | Step:679ms | Loss: 1.7556
[15300/72000 (Shard 300/3000 |  21%)] LR:2.81e-04 Fwd:3ms Loss:157ms Opt:196ms | Step:545ms | Loss: 1.7165 | Val: 1.7016
[15600/72000 (Shard 600/3000 |  21%)] LR:2.80e-04 Fwd:3ms Loss:157ms Opt:196ms | Step:546ms | Loss: 1.7028 | Val: 1.7002
[15900/72000 (Shard 900/3000 |  22%)] LR:2.79e-04 Fwd:3ms Loss:157ms Opt:194ms | Step:546ms | Loss: 1.7302 | Val: 1.7598
[16200/72000 (Shard 1200/3000 |  22%)] LR:2.78e-04 Fwd:3ms Loss:157ms Opt:196ms | Step:546ms | Loss: 1.8169 | Val: 1.7130
[16500/72000 (Shard 1500/3000 |  22%)] LR:2.77e-04 Fwd:3ms Loss:157ms Opt:194ms | Step:545ms | Loss: 1.7610 | Val: 1.8143
[16800/72000 (Shard 1800/3000 |  23%)] LR:2.76e-04 Fwd:3ms Loss:157ms Opt:196ms | Step:546ms | Loss: 1.7516 | Val: 1.8470
[17100/72000 (Shard 2100/3000 |  23%)] LR:2.75e-04 Fwd:3ms Loss:157ms Opt:195ms | Step:546ms | Loss: 1.8572 | Val: 1.8443
[17400/72000 (Shard 2400/3000 |  24%)] LR:2.74e-04 Fwd:3ms Loss:157ms Opt:196ms | Step:546ms | Loss: 1.7850 | Val: 1.8959
[17700/72000 (Shard 2700/3000 |  24%)] LR:2.73e-04 Fwd:3ms Loss:157ms Opt:194ms | Step:546ms | Loss: 1.8910 | Val: 1.9042
[18000/72000 (Shard 3000/3000 |  25%)] LR:2.72e-04 Fwd:3ms Loss:157ms Opt:196ms | Step:546ms | Loss: 2.0018 | Val: 1.9835

------------------------------------------------------------
[Shard Index 07 (8/24)] Input: shard_07.txt -> Saving Checkpoint: tinystories_shard07.bin
------------------------------------------------------------
Copying previous checkpoint (tinystories_shard06.bin -> tinystories_shard07.bin) to resume state...
Running: .\build\Release\mgpt.exe -t -g -d="C:\Users\bogaz\MGPT\mgpt\tinystories\shard_07.txt" -f="tinystories_shard07.bin" --lr=0.0003 -s=3000 --total-steps=72000 -l=12 -c=384 -w=256 -b=16 -a=2 --vocab-file="tinystories_slice.txt.vocab.bin" --resume
============================================================
      MGPT BOOTING UP...
============================================================

Using dataset at: C:\Users\bogaz\MGPT\mgpt\tinystories\shard_07.txt
Found cached binary dataset files:
      -> Loaded Vocabulary Size = 4096 tokens (tinystories_slice.txt.vocab.bin)
      -> Loading pre-encoded token stream from C:\Users\bogaz\MGPT\mgpt\tinystories\shard_07.txt.tok.bin...
      Total Dataset Tokens: 24125355 (Loaded instantly from binary cache!)
Train/Val Split (90/10):
      -> Training Tokens:   21712819
      -> Validation Tokens: 2412536

Instantiated GPT Model Architecture:
      -> Vocab Size:    4096
      -> Max Seq Len:   256
      -> Embed Dim:     384
      -> Num Layers:    12
      -> Checkpointing: ENABLED (Block-Level Activation Recomputation)
      -> Total Params:  31531264 float32 parameters (~123169 KB)

Migrating GPT Model and Engine to CUDA GPU...
Resuming Training Mode: Loading existing checkpoint from tinystories_shard07.bin...
Successfully loaded GPT model weights from tinystories_shard07.bin (restored at completed step 18000)!
Uploading Train and Validation Datasets to GPU Memory...
Successfully loaded AdamW optimizer state (step 6000) from tinystories_shard07.bin.opt!
Starting Training Loop (AdamW, LR=0.0003, Micro-Batch=16, Accum Steps=2 [Effective Batch: 32], Shard Steps=3000, Global Horizon=18000 -> 21000/72000)...
------------------------------------------------------------
Step       Progress & Timings                      Train Loss   Val Loss
------------------------------------------------------------
[18001/72000 (Shard 1/3000 |  25%)] LR:2.72e-04 Fwd:183ms Loss:120ms Opt:194ms | Step:676ms | Loss: 1.9882
[18300/72000 (Shard 300/3000 |  25%)] LR:2.70e-04 Fwd:3ms Loss:156ms Opt:195ms | Step:546ms | Loss: 2.1463 | Val: 2.0813
[18600/72000 (Shard 600/3000 |  25%)] LR:2.69e-04 Fwd:3ms Loss:156ms Opt:196ms | Step:546ms | Loss: 2.1463 | Val: 2.1657
[18900/72000 (Shard 900/3000 |  26%)] LR:2.68e-04 Fwd:3ms Loss:156ms Opt:195ms | Step:546ms | Loss: 2.2471 | Val: 2.2751
[19200/72000 (Shard 1200/3000 |  26%)] LR:2.67e-04 Fwd:3ms Loss:158ms Opt:194ms | Step:543ms | Loss: nan
PS C:\Users\bogaz\MGPT\mgpt>

========================

sing dataset at: C:\Users\bogaz\MGPT\mgpt\tinystories\shard_03.txt
Found cached binary dataset files:
      -> Loaded Vocabulary Size = 4096 tokens (tinystories_slice.txt.vocab.bin)
      -> Loading pre-encoded token stream from C:\Users\bogaz\MGPT\mgpt\tinystories\shard_03.txt.tok.bin...
      Total Dataset Tokens: 24135917 (Loaded instantly from binary cache!)
Train/Val Split (90/10):
      -> Training Tokens:   21722325
      -> Validation Tokens: 2413592

Instantiated GPT Model Architecture:
      -> Vocab Size:    4096
      -> Max Seq Len:   256
      -> Embed Dim:     384
      -> Num Layers:    12
      -> Checkpointing: ENABLED (Block-Level Activation Recomputation)
      -> Total Params:  31531264 float32 parameters (~123169 KB)

Migrating GPT Model and Engine to CUDA GPU...
Resuming Training Mode: Loading existing checkpoint from tinystories_shard03.bin...
Successfully loaded GPT model weights from tinystories_shard03.bin (restored at completed step 9000)!
Uploading Train and Validation Datasets to GPU Memory...
Successfully loaded AdamW optimizer state (step 9000) from tinystories_shard03.bin.opt!
Starting Training Loop (AdamW, LR=0.0003, Micro-Batch=16, Accum Steps=2 [Effective Batch: 32], Shard Steps=3000, Global Horizon=9000 -> 12000/72000)...
------------------------------------------------------------
Step       Progress & Timings                      Train Loss   Val Loss
------------------------------------------------------------
[9001/72000 (Shard 1/3000 |  12%)] LR:2.96e-04 Fwd:183ms Loss:117ms Opt:195ms | Step:669ms | Loss: 1.7823
[9300/72000 (Shard 300/3000 |  12%)] LR:2.95e-04 Fwd:3ms Loss:157ms Opt:199ms | Step:550ms | Loss: 1.7413 | Val: 1.6970
[9600/72000 (Shard 600/3000 |  13%)] LR:2.95e-04 Fwd:3ms Loss:159ms Opt:195ms | Step:548ms | Loss: 1.6358 | Val: 1.7339
[9900/72000 (Shard 900/3000 |  13%)] LR:2.94e-04 Fwd:3ms Loss:158ms Opt:197ms | Step:549ms | Loss: 1.7579 | Val: 1.7180
[10200/72000 (Shard 1200/3000 |  14%)] LR:2.94e-04 Fwd:3ms Loss:157ms Opt:196ms | Step:548ms | Loss: 1.6853 | Val: 1.6817
[10500/72000 (Shard 1500/3000 |  14%)] LR:2.93e-04 Fwd:3ms Loss:157ms Opt:196ms | Step:547ms | Loss: 1.6549 | Val: 1.6811
[10800/72000 (Shard 1800/3000 |  15%)] LR:2.93e-04 Fwd:3ms Loss:159ms Opt:195ms | Step:547ms | Loss: 1.6248 | Val: 1.7779
[11100/72000 (Shard 2100/3000 |  15%)] LR:2.92e-04 Fwd:3ms Loss:158ms Opt:195ms | Step:545ms | Loss: nan
[11400/72000 (Shard 2400/3000 |  15%)] LR:2.91e-04 Fwd:3ms Loss:157ms Opt:196ms | Step:544ms | Loss: nan
[11700/72000 (Shard 2700/3000 |  16%)] LR:2.91e-04 Fwd:3ms Loss:157ms Opt:196ms | Step:545ms | Loss: nan
[12000/72000 (Shard 3000/3000 |  16%)] LR:2.90e-04 Fwd:3ms Loss:158ms Opt:195ms | Step:545ms | Loss: nan


============================================================
      MGPT BOOTING UP...
============================================================

Using dataset at: C:\Users\bogaz\MGPT\mgpt\tinystories\shard_04.txt
Found cached binary dataset files:
      -> Loaded Vocabulary Size = 4096 tokens (tinystories_slice.txt.vocab.bin)
      -> Loading pre-encoded token stream from C:\Users\bogaz\MGPT\mgpt\tinystories\shard_04.txt.tok.bin...
      Total Dataset Tokens: 24132786 (Loaded instantly from binary cache!)
Train/Val Split (90/10):
      -> Training Tokens:   21719507
      -> Validation Tokens: 2413279

Instantiated GPT Model Architecture:
      -> Vocab Size:    4096
      -> Max Seq Len:   256
      -> Embed Dim:     384
      -> Num Layers:    12
      -> Checkpointing: ENABLED (Block-Level Activation Recomputation)
      -> Total Params:  31531264 float32 parameters (~123169 KB)

Migrating GPT Model and Engine to CUDA GPU...
Resuming Training Mode: Loading existing checkpoint from tinystories_shard04.bin...
Successfully loaded GPT model weights from tinystories_shard04.bin (restored at completed step 12000)!
Uploading Train and Validation Datasets to GPU Memory...
Successfully loaded AdamW optimizer state (step 12000) from tinystories_shard04.bin.opt!
Starting Training Loop (AdamW, LR=0.0003, Micro-Batch=16, Accum Steps=2 [Effective Batch: 32], Shard Steps=3000, Global Horizon=12000 -> 15000/72000)...
------------------------------------------------------------
Step       Progress & Timings                      Train Loss   Val Loss
------------------------------------------------------------
[12001/72000 (Shard 1/3000 |  16%)] LR:2.90e-04 Fwd:179ms Loss:119ms Opt:194ms | Step:665ms | Loss: nan
[12300/72000 (Shard 300/3000 |  17%)] LR:2.89e-04 Fwd:3ms Loss:157ms Opt:195ms | Step:545ms | Loss: nan
[12600/72000 (Shard 600/3000 |  17%)] LR:2.89e-04 Fwd:3ms Loss:159ms Opt:195ms | Step:545ms | Loss: nan
[12900/72000 (Shard 900/3000 |  17%)] LR:2.88e-04 Fwd:3ms Loss:158ms Opt:196ms | Step:545ms | Loss: nan
[13200/72000 (Shard 1200/3000 |  18%)] LR:2.87e-04 Fwd:3ms Loss:159ms Opt:195ms | Step:545ms | Loss: nan
[13500/72000 (Shard 1500/3000 |  18%)] LR:2.86e-04 Fwd:3ms Loss:158ms Opt:195ms | Step:545ms | Loss: nan
[13800/72000 (Shard 1800/3000 |  19%)] LR:2.85e-04 Fwd:3ms Loss:158ms Opt:195ms | Step:545ms | Loss: nan
[14100/72000 (Shard 2100/3000 |  19%)] LR:2.85e-04 Fwd:3ms Loss:158ms Opt:196ms | Step:545ms | Loss: nan
[14400/72000 (Shard 2400/3000 |  20%)] LR:2.84e-04 Fwd:3ms Loss:158ms Opt:196ms | Step:545ms | Loss: nan

========================

      -> Loaded Vocabulary Size = 4096 tokens (tinystories_slice.txt.vocab.bin)
      -> Loading pre-encoded token stream from C:\Users\bogaz\MGPT\mgpt\tinystories\shard_02.txt.tok.bin...
      Total Dataset Tokens: 24136041 (Loaded instantly from binary cache!)
Train/Val Split (90/10):
      -> Training Tokens:   21722436
      -> Validation Tokens: 2413605

Instantiated GPT Model Architecture:
      -> Vocab Size:    4096
      -> Max Seq Len:   256
      -> Embed Dim:     384
      -> Num Layers:    12
      -> Checkpointing: ENABLED (Block-Level Activation Recomputation)
      -> Total Params:  31531264 float32 parameters (~123169 KB)

Migrating GPT Model and Engine to CUDA GPU...
Resuming Training Mode: Loading existing checkpoint from tinystories_shard02.bin...
Successfully loaded GPT model weights from tinystories_shard02.bin (restored at completed step 9000)!
Uploading Train and Validation Datasets to GPU Memory...
Successfully loaded AdamW optimizer state (step 9000) from tinystories_shard02.bin.opt!
Starting Training Loop (AdamW, LR=0.0003, Micro-Batch=16, Accum Steps=2 [Effective Batch: 32], Shard Steps=3000, Global Horizon=9000 -> 12000/72000)...
------------------------------------------------------------
Step       Progress & Timings                      Train Loss   Val Loss
------------------------------------------------------------
[9001/72000 (Shard 1/3000 |  12%)] LR:2.96e-04 Fwd:178ms Loss:117ms Opt:194ms | Step:664ms | Loss: 1.6668
[9300/72000 (Shard 300/3000 |  12%)] LR:2.95e-04 Fwd:3ms Loss:158ms Opt:195ms | Step:548ms | Loss: 1.7253 | Val: 1.7501
[9600/72000 (Shard 600/3000 |  13%)] LR:2.95e-04 Fwd:3ms Loss:157ms Opt:195ms | Step:547ms | Loss: 1.6063 | Val: 1.7251
[9900/72000 (Shard 900/3000 |  13%)] LR:2.94e-04 Fwd:3ms Loss:159ms Opt:195ms | Step:547ms | Loss: 1.5832 | Val: 1.7164
[10200/72000 (Shard 1200/3000 |  14%)] LR:2.94e-04 Fwd:3ms Loss:157ms Opt:197ms | Step:547ms | Loss: 1.4897 | Val: 1.7678
[10500/72000 (Shard 1500/3000 |  14%)] LR:2.93e-04 Fwd:3ms Loss:157ms Opt:197ms | Step:547ms | Loss: 1.5506 | Val: 1.7214
[10800/72000 (Shard 1800/3000 |  15%)] LR:2.93e-04 Fwd:3ms Loss:157ms Opt:195ms | Step:548ms | Loss: 1.7068 | Val: 1.9348
[11100/72000 (Shard 2100/3000 |  15%)] LR:2.92e-04 Fwd:3ms Loss:159ms Opt:195ms | Step:545ms | Loss: nan
PS C:\Users\bogaz\MGPT\mgpt>


# NEXT STEPS & TODO:
- Implement dataset sharding / binary shard loading (`.bin` files) for large datasets (2GB+ TinyStories) with invisible model continuation / `--resume` checkpoints.
- Build Goku roleplay and Text-to-SQL commercial milestone bots.
