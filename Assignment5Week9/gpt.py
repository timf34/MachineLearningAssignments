from datetime import datetime
import os
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
print("using: ", device)
eval_iters = 200

n_head = 4

n_embd = 64
n_layer = 4

dropout = 0.2
# ------------

torch.manual_seed(1337)

# Load datasets
datasets = {
    "childSpeech_training": "input_childSpeech_trainingSet.txt",
    "childSpeech_test": "input_childSpeech_testSet.txt",
    "shakespeare": "input_shakespeare.txt"
}

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open(datasets["childSpeech_training"], 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, use_bias=False):  # Add use_bias parameter
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=use_bias)
        self.query = nn.Linear(n_embd, head_size, bias=use_bias)
        self.value = nn.Linear(n_embd, head_size, bias=use_bias)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, use_bias=False):  # Add use_bias parameter
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, use_bias) for _ in range(num_heads)])
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

    def __init__(self, n_embd, n_head, use_bias=False, use_skip=True):  # Add use_skip parameter
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, use_bias)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.use_skip = use_skip

    def forward(self, x):
        if self.use_skip:
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        else:
            x = self.sa(self.ln1(x))
            x = self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, use_bias=False, use_skip=True):  # Add use_skip parameter
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, use_bias=use_bias, use_skip=use_skip) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
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

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


def model_training(use_bias=False, use_skip=True):
    model = GPTLanguageModel(use_bias=use_bias, use_skip=use_skip)
    m = model.to(device)
    print(
        f"Training model with {'bias' if use_bias else 'no bias'} in attention layers and {'skip connections' if use_skip else 'no skip connections'}")
    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

    with open('training_results.txt', 'a') as f:
        f.write(f"\nModel with bias={use_bias}, skip={use_skip}:\n")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            with open('training_results.txt', 'a') as f:
                f.write(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()



    # Save weights for model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = 'model_weights'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'model_weights_{timestamp}.pt')
    torch.save(m.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    #open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))


def analyze_dataset(name, text):
    unique_chars = sorted(set(text))
    vocab_size = len(unique_chars)
    length = len(text)
    # Word analysis
    words = text.split()
    unique_words = set(words)
    sentences = text.split('\n')
    non_empty_sentences = [s for s in sentences if s.strip()]

    print(f"Dataset: {name}")
    print(f" - Vocabulary size (chars): {vocab_size}")
    print(f" - Total length: {length} characters")
    print(f" - Unique words: {len(unique_words)}")
    print(f" - Total words: {len(words)}")
    print(f" - Number of lines: {len(non_empty_sentences)}")
    print(f" - Average words per line: {len(words) / len(non_empty_sentences):.2f}")
    print(f" - Sample (first 100 characters):\n{text[:100]}")
    print("-" * 40)


def main():
    # dataset_contents = {}
    # for name, file_path in datasets.items():
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         dataset_contents[name] = f.read()
    #
    # # Analyze datasets
    # for name, text in dataset_contents.items():
    #     analyze_dataset
    pass


def evaluate_model_on_test_set(model, test_file_path, model_weights_path, baseline=False):
    """Evaluate the given model on the test set."""
    # Load the test set
    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_text = f.read()

    # Encode the test set
    test_data = torch.tensor(encode(test_text), dtype=torch.long)
    test_data = test_data.to(device)

    # Prepare the model
    if not baseline:
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        model.eval()  # Set the model to evaluation mode
        print(f"Model weights loaded from {model_weights_path}")

    # Compute loss on the test set
    batch_size = 64  # Adjust batch size if necessary
    block_size = 256  # Same as during training
    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        for i in range(0, len(test_data) - block_size, batch_size):
            inputs = []
            targets = []
            for j in range(batch_size):
                if i + j + block_size >= len(test_data):
                    break
                inputs.append(test_data[i + j:i + j + block_size])
                targets.append(test_data[i + j + 1:i + j + block_size + 1])

            if not inputs:
                break

            inputs = torch.stack(inputs).to(device)
            targets = torch.stack(targets).to(device)

            logits, loss = model(inputs, targets)
            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss


def create_baseline_model():
    """Create a dummy/baseline model that predicts uniform probabilities."""

    class BaselineModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.uniform_probs = nn.Parameter(torch.ones(vocab_size), requires_grad=False)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            logits = self.uniform_probs.unsqueeze(0).unsqueeze(0).repeat(B, T, 1)
            if targets is None:
                loss = None
            else:
                logits = logits.view(-1, vocab_size)
                targets = targets.view(-1)
                loss = F.cross_entropy(logits, targets)
            return logits, loss

    return BaselineModel().to(device)


def main_evaluation():
    # Load the trained model
    trained_model = GPTLanguageModel().to(device)
    test_file_path = datasets["childSpeech_test"]
    model_weights_path = "model_weights/model_weights_20241207_224359.pt"

    print("Evaluating trained model:")
    trained_model_loss = evaluate_model_on_test_set(trained_model, test_file_path, model_weights_path)

    print("\nEvaluating baseline model:")
    baseline_model = create_baseline_model()
    baseline_model_loss = evaluate_model_on_test_set(baseline_model, test_file_path, model_weights_path, baseline=True)

    print("\nComparison:")
    print(f"Trained Model Loss: {trained_model_loss:.4f}")
    print(f"Baseline Model Loss: {baseline_model_loss:.4f}")

    if trained_model_loss < baseline_model_loss:
        print("The trained model performs better than the baseline.")
    else:
        print("The baseline performs comparably or better than the trained model.")


if __name__ == "__main__":
    main_evaluation()


if __name__ == "__main__":
    # https://claude.ai/chat/528a7ac1-7386-4f97-a4d8-c28d54cc2468
    # print("Training model without bias terms:")
    # model_training(use_bias=False)
    # print("\nTraining model with bias terms:")
    # model_training(use_bias=True)
    # use_bias = [False, True]
    # use_skip = [False, True]
    #
    # for i in use_bias:
    #     for j in use_skip:
    #         print(f"Training model with: \nBias: {i} \nSkip: {j}")
    #         model_training(use_bias=i, use_skip=j)
    main_evaluation()


