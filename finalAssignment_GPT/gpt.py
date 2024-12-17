import datetime
import math
import os  # Added for directory management
import json  # Added for saving losses
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.3  # Increased dropout from 0.2 to 0.3
weight_decay = 1e-5  # Added weight decay for regularization

torch.manual_seed(1337)

# ------------------ DATASET LOADING CHANGES -------------------
# Instead of loading a text dataset like Shakespeare, we load the melody dataset.
# The dataset: inputMelodiesAugmented.txt (already included in the provided dataset folder)

with open('finalAssignment_musicDataset/inputMelodiesAugmented.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# The text now consists of sequences of melodic tokens (notes and rests).
# Each character (e.g., 'C', 'c', 'D', 'R', ' ', etc.) is considered a token.
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ------------------ DATA AUGMENTATION: TRANSPOSE FUNCTION -------------------
# def transpose_batch(x, shift, stoi, itos, note_tokens):
#     """
#     Transpose the batch by shifting note tokens.
#
#     Parameters:
#     - x: Tensor of shape (batch_size, block_size)
#     - shift: Integer, number of semitones to shift (positive or negative)
#     - stoi: Dictionary mapping characters to indices
#     - itos: Dictionary mapping indices to characters
#     - note_tokens: Set of tokens representing notes
#
#     Returns:
#     - Transposed tensor
#     """
#     x_transposed = x.clone()
#     for i in range(x.size(0)):  # Iterate over batch
#         for j in range(x.size(1)):  # Iterate over sequence
#             token = itos[x[i, j].item()]
#             if token in note_tokens:
#                 # Find current position in note_tokens
#                 current_idx = note_tokens_sorted.index(token)
#                 # Compute new index with shift and wrap around
#                 new_idx = (current_idx + shift) % len(note_tokens_sorted)
#                 new_token = note_tokens_sorted[new_idx]
#                 x_transposed[i, j] = stoi[new_token]
#     return x_transposed
#
# # Define note tokens (excluding rests)
# # Adjust this set based on your actual note tokens
# note_tokens = set(chars) - set(['R', ' '])  # Assuming 'R' is rest and ' ' is space
# note_tokens_sorted = sorted(list(note_tokens))  # Sorted list for consistent shifting

def get_batch(split):
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_) - block_size, (batch_size,))
    x = torch.stack([data_[i:i + block_size] for i in ix])
    y = torch.stack([data_[i + 1:i + block_size + 1] for i in ix])

    # ------------------ DATA AUGMENTATION: RANDOM TRANSPOSE -------------------
    # With a probability of 0.5, transpose the batch by a random shift between -2 and +2 semitones
    # if split == 'train' and torch.rand(1).item() < 0.5:
    #     shift = torch.randint(-2, 3, (1,)).item()  # Random shift: -2, -1, 0, 1, 2
    #     if shift != 0:
    #         print(f"Transposing batch by {shift} semitone(s).")
    #         x = transpose_batch(x, shift, stoi, itos, note_tokens)

    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        avg_loss = losses.mean().item()
        # Compute perplexity = exp(cross_entropy)
        perplexity = math.exp(avg_loss)
        out[split] = {'loss': avg_loss, 'perplexity': perplexity}
    model.train()
    return out

def compute_baseline_perplexity():
    """
    Computes the perplexity of a simple baseline model that predicts uniformly
    at random from the vocabulary. The baseline does not depend on context.
    It simply guesses each next token with probability = 1/vocab_size.
    """
    # We'll evaluate on the validation set
    # The cross-entropy for a uniform distribution over vocab_size for a given
    # target token is -log(1/vocab_size) = log(vocab_size).
    # If predictions are uniform random, cross-entropy is just log(vocab_size).
    # Perplexity for uniform guesser is just vocab_size.

    # However, let's confirm this by directly computing the average negative log-likelihood:
    # The probability assigned to the correct token = 1/vocab_size
    # negative log likelihood for each token = -log(1/vocab_size) = log(vocab_size)
    # so cross-entropy = log(vocab_size), perplexity = exp(log(vocab_size)) = vocab_size.

    # Direct calculation:
    baseline_loss = math.log(vocab_size)  # since cross entropy = log(vocab_size)
    baseline_perplexity = vocab_size  # perplexity = exp(log(vocab_size)) = vocab_size
    return baseline_loss, baseline_perplexity

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Initialize the model and move it to the appropriate device
model = GPTLanguageModel().to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

# Create a PyTorch optimizer with weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Create a learning rate scheduler that reduces LR when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# Compute and print baseline perplexity
baseline_loss, baseline_pp = compute_baseline_perplexity()
print(f"Baseline (random) cross-entropy: {baseline_loss:.4f}, perplexity: {baseline_pp:.4f}")

# Initialize lists to store loss metrics
training_losses = []
validation_losses = []
training_perplexities = []
validation_perplexities = []

# ------------------ Setup for Saving Models and Losses -------------------
# Define a directory to save model checkpoints with a unique timestamp
current_time = datetime.datetime.now().strftime("%H_%M_%d_%m_%Y")  # hours_minutes_date_month_year
checkpoint_dir = os.path.join('checkpoints', current_time)
os.makedirs(checkpoint_dir, exist_ok=True)

print(f"Model checkpoints will be saved in: {checkpoint_dir}")

# ------------------ Training Loop -------------------
for iter in range(max_iters):
    # Evaluate the model periodically
    if iter % eval_interval == 0 or iter == max_iters - 1:
        metrics = estimate_loss()
        train_loss = metrics['train']['loss']
        val_loss = metrics['val']['loss']
        train_pp = metrics['train']['perplexity']
        val_pp = metrics['val']['perplexity']
        print(
            f"step {iter}: train loss {train_loss:.4f}, train perplexity {train_pp:.4f}, "
            f"val loss {val_loss:.4f}, val perplexity {val_pp:.4f}"
        )

        # Append metrics to the lists
        training_losses.append({'iter': iter, 'loss': train_loss})
        validation_losses.append({'iter': iter, 'loss': val_loss})
        training_perplexities.append({'iter': iter, 'perplexity': train_pp})
        validation_perplexities.append({'iter': iter, 'perplexity': val_pp})

        # Save the model checkpoint in the unique timestamped directory
        checkpoint_path = os.path.join(checkpoint_dir, f'iters-{iter}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint to {checkpoint_path}")

        # Update the learning rate scheduler based on validation loss
        scheduler.step(val_loss)

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ------------------ Save Loss Metrics -------------------
# Combine all loss metrics into a single dictionary
loss_metrics = {
    'training_losses': training_losses,
    'validation_losses': validation_losses,
    'training_perplexities': training_perplexities,
    'validation_perplexities': validation_perplexities
}

# Define the path to save the loss metrics in the unique directory
loss_metrics_path = os.path.join(checkpoint_dir, 'losses.json')

# Save the loss metrics to a JSON file
with open(loss_metrics_path, 'w') as f:
    json.dump(loss_metrics, f, indent=4)

print(f"Saved loss metrics to {loss_metrics_path}")

# ------------------ Generate a Short Melody Sample -------------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_sequence = model.generate(context, max_new_tokens=200)[0].tolist()
generated_melody = decode(generated_sequence)
print("Generated Melody:\n", generated_melody)
