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