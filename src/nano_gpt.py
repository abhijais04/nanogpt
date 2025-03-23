import torch
import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
block_size = 128 # Context length
batch_size = 64 # Batches of input getting processed in parallel
max_iters = 1000 # Number of training iterations
learning_rate = 3e-4 
device = "cuda" if torch.cuda.is_available() else "cpu"
n_layer = 6 # number of layers in the GPT, or number of blocks
n_embd = 32 # embedding dimention of the token
n_head = 4 # number of heads in a single block
# Turn off x% of random nodes during forward and backward pass to avoid overfitting
dropout = 0.2
# ----------------------------

# These don't have any impact on GPT architecture
train_split = 0.9
eval_interval = 500 # Interval during training for doing evaluation
eval_iterations = 200 # number of different samples for evaluating model performance
# -----------------------

torch.manual_seed(1337)


with open("../data/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# unique characters in the data
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Creating functions for encoding/decoding character
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda input_string: [stoi[c] for c in input_string]
decode = lambda int_array: "".join([itos[i] for i in int_array])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
train_data = data[: int(train_split * len(data))]
validation_data = data[int(train_split * len(data)) :]


# Get random batches of data each with block-size length
def get_batch(split):
    tmp_data = train_data if split == "train" else validation_data
    initial_pos_for_random_batch = torch.randint(
        0, len(tmp_data) - block_size, (batch_size,)
    )
    x = torch.stack(
        [tmp_data[i : i + block_size] for i in initial_pos_for_random_batch]
    )
    y = torch.stack(
        [tmp_data[i + 1 : i + block_size + 1] for i in initial_pos_for_random_batch]
    )
    x, y = x.to(device), y.to(device)
    return x, y

# Estimating training and validation loss for currently trained models
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):
            X, Y = get_batch(split=split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

"""
Single head in a layer
"""
class Head(nn.Module):
    # head_size is representing the dimentionality of Key and query space.
    def __init__(self, head_size):
        super().__init__()
        # Key matrix
        self.key = nn.Linear(n_embd, head_size, bias=False)
        # Query matrix
        self.query = nn.Linear(n_embd, head_size, bias=False)
        # Value matrix
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # Calculate keys for input tokens
        k = self.key(x) # (B, T, head_size)
        # Calculate queries for input tokens
        q = self.query(x) # (B, T, head_size)

        # compute attention score ("affinities")
        # divide by square root of embedding size to keep weights distributed
        wei = (
            q @ k.transpose(-2, -1) * (C**0.5)
        )  # (B,T,head_size) @ (B, head_size, T) ---> (B, T, T)

        """
        wei is the T * T matrix, which wei[i][j] shows affinity of ith token coming from jth token 
        in the input. We do not want any affinity from any token Y which is coming after a token X.
        So wei[i][j] should be zero when i < j. This is achieved by a upper triangular matrix masking.
        Before taking softmax, we make it -infinity so after softmax it becomes zero.
        """
       
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)

        """
        After taking softmax, half of the matrix (right-upper half)  becomes 0.
        And for each row sum of all numbers will be 1, and they will be between 0 and 1
        Now the weight matrix is a properly normalised
        """

        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # Turn off random weights.
        wei = self.dropout(wei) # (B, T, T)

        # Convert token embedding to a value embedding.
        # In this case, size of value embedding in head_size
        # because this is from a single head, and we will concatenate
        # outputs from all heads. But this can change based on algorithm
        v = self.value(x)  # (B, T, C) --->  #(B, T, head_size)


        """
        For each token X in Token list T, Take weighted sum of values embedding 
        of all tokens coming before X in the input. To ensure this, we had done 
        matrix masking earlier. Weights are calculatd by key and query matrix
        """
        
        out = wei @ v  # (B, T, T) @ (B, T, head_size) ----> (B, T, head_size)

        """
        This output should be added into the original token to make the token 
        context rich
        """
        return out

"""
Multihead module, runs n heads in parallel for aggregation of attention
"""
class MultiHeadAttention(nn.Module):
    """ Multiple heads of self attentions in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # concetenate weights from all heads 
        # (B, T, n_head * head_size) => (B, T, n_embd) or (B, T, C)
        out = torch.cat([h(x) for h in self.heads], dim=-1) 

        out = self.dropout(self.proj(out)) # (num_heads*B, T, C)

        return out

"""
Simple Multilayer perceptron with dropout
"""
class FeedForwardLayer(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

"""
Single block in the GPT
""" 
class Block(nn.Module):
    """Transformer block: Communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        """
        Output from a single head is of size (B, T, head_size), and outputs from all heads 
        are concatenated so output dimention after that becomes (B, T, head_size * num_head)
        We need the output in (B, T, n_embd). So we create the number of heads in such a way 
        n_embd = head_size * num_heads
        """
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(num_heads=n_head, head_size=head_size)
        self.ffwd = FeedForwardLayer(n_embd=n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Each token directly reads the logits for the next token from lookup table

        # random embedding_table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # metrix to represent positional value of tokens
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd=n_embd, n_head=n_head) for _ in range(n_layer)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(n_embd)

        # Matrix for converting a single character of dimestion n to a
        # probability matrix for each character in vocab.
        self.lm_head = nn.Linear(n_embd, vocab_size)

    # idx is B*T table where each item represents a token.
    def forward(self, idx, targets=None):

        B, T = idx.shape
        # C is embedding dimention = n_embd
        # B -> number of batches running in parallel
        # T is number of tokens = block_size

        token_embeddings = self.token_embedding_table(idx)  # (B, T, C)

        pos_embeddings = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)

        x = token_embeddings + pos_embeddings  # (B, T, C)

        x = self.blocks(x)  # (B, T, C)

        x = self.ln_f(x)  # (B, T, C)

        # (Batch, Block, output) or (B, T, C) or (4, 8, 65)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Loss function calculation
            # pytorch expects (B*T, C) in input
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in current context

        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # get predictions
            logits, loss = self(idx_cond)

            # logit have dimention of (B, T, C)
            # focus only on the last time step, or the last token
            logits = logits[:, -1, :]  # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


# Get a random batch
xb, yb = get_batch("train")
model = BigramLanguageModel()
m = model.to(device)

optimiser = torch.optim.AdamW(m.parameters(), lr=learning_rate)


# train the model
for steps in range(max_iters):

    # get the losses of currently trained data
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {steps}: train loss {losses['train']:0.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = m(xb, yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()

# generate output from model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

torch.save(m.state_dict(), "../trained_models/128_64_V1_weights.pth")
