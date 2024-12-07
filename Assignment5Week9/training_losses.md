# Training losses 

Will note down the training losses for different configs here when training on the ChildSpeech training set. 

### Config 1

**The config:**
```python
# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2
```


**The output (incl. losses):**
```bash 
using:  cuda
0.834856 M parameters
step 0: train loss 3.7909, val loss 3.7915
step 500: train loss 0.3599, val loss 0.3621
step 1000: train loss 0.3403, val loss 0.3437
step 1500: train loss 0.3343, val loss 0.3371
step 2000: train loss 0.3312, val loss 0.3354
step 2500: train loss 0.3303, val loss 0.3340
step 3000: train loss 0.3295, val loss 0.3347
step 3500: train loss 0.3286, val loss 0.3349
step 4000: train loss 0.3272, val loss 0.3348
step 4500: train loss 0.3263, val loss 0.3351
step 4999: train loss 0.3260, val loss 0.3352

No nap I did it
I wash hands I sing
Daddy play Look airplane
Big hug Where kitty go
Shoes on I run fast
Look moon Yummy apple
Saw big fluffy doggy at park it run fast Read book
I found it No like it
I seee bird I love you
I want that Help me please
No nap Read book
Read book Uh oh spill
I want cookie Come here
I tired Big truck
All done I want cookie
I love you Uh oh spill
Saw big fluffy doggy at park it run fast Where ball
Saw big fluffy doggy at park it run fast Night night
I sing Read book
I 
```

### Config 2 

The config:
```python
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200

n_embd = 256
n_head = 4
n_layer = 1

dropout = 0.2
```

The output (incl. losses):
```bash
using:  cuda
0.87556 M parameters
step 0: train loss 3.7483, val loss 3.7492
step 500: train loss 0.3491, val loss 0.3552
step 1000: train loss 0.3413, val loss 0.3478
step 1500: train loss 0.3357, val loss 0.3438
step 2000: train loss 0.3364, val loss 0.3450
step 2500: train loss 0.3342, val loss 0.3441
step 3000: train loss 0.3332, val loss 0.3450
step 3500: train loss 0.3302, val loss 0.3442
step 4000: train loss 0.3288, val loss 0.3444
step 4500: train loss 0.3267, val loss 0.3449
step 4999: train loss 0.3253, val loss 0.3453
Model weights saved to model_weights/model_weights_20241207_223612.pt

What is that What is that
Where kitty go I love you
Shoes on All done
I draw I climb
Mama I want play with big red ball outside More bubbles
Where ball I draw
No stop My teddy
I draw I see doggy
More bubbles I jump
Bath time I hide
Mama I want play with big red ball outside
No line Mo teddy I want corkie
Big hug fy jund ice plleay
What is thath I climb
I tin hthat Go park
Datboready t read meakine
I hids idonAll d gyone
Mo t more He me p please
I jump I want that
I lire I hungum
Wh t that is t w
```

### Config 3

The config:
```python
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("using: ", device)
eval_iters = 200

n_embd = 64
n_head = 4
n_layer = 4

dropout = 0.2
```

The output (incl. losses):
```bash
using:  cuda
0.22084 M parameters
step 0: train loss 3.6824, val loss 3.6835
step 500: train loss 0.5153, val loss 0.5194
step 1000: train loss 0.3515, val loss 0.3528
step 1500: train loss 0.3408, val loss 0.3430
step 2000: train loss 0.3369, val loss 0.3389
step 2500: train loss 0.3343, val loss 0.3373
step 3000: train loss 0.3331, val loss 0.3361
step 3500: train loss 0.3305, val loss 0.3341
step 4000: train loss 0.3309, val loss 0.3343
step 4500: train loss 0.3297, val loss 0.3333
step 4999: train loss 0.3299, val loss 0.3338
Model weights saved to model_weights/model_weights_20241207_224359.pt

Mout bbes Where ball
I want that My teddy
I want that I tired
I draw I found it
I tired Where ball
I love you Night night
Daddy read me dinosaur book before bed I see doggy
Yummy apple I hide
No stop What is that
I make mess I did it
Help me please I did it
All gone I jump high
I want cookie All done
I want thatvi Go park
What is that Daddy read me dinosaur book before bed
Bath top I want that
I want cookie Go park
Help me please Help me please
Yummy apple No stop
All done I did it
I see doggy W
```