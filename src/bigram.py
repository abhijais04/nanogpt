# -*- coding: utf-8 -*-
import torch
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# hyperparameters
block_size = 8
batch_size = 32
train_split = 0.9
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iterations = 200
n_embd = 32
# ----------------------------

with open("data/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# unique characters in the data
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Creating functions for encoding decodeing
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda input_string: [stoi[c] for c in input_string]
decode = lambda int_array: "".join([itos[i] for i in int_array])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
train_data = data[: int(train_split * len(data))]
validation_data = data[int(train_split * len(data)) :]


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


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Each token directly reads the logits for the next token from lookup table

        # random embedding_table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Matrix for converting a single character of dimestion n to a
        # probability metrics for each character in vocab.
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # metrix to represent positional value of tokens
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

    # idx is B*T table where each item represents a token.
    def forward(self, idx, targets=None):
        """
        This is a pretty simple way for calculating the forward pass
        without hidden layers.
        In this simple bigram model, each token's prediction is
        based only on the previous token (hence bigram). The
        embeddings capture the "meaning" of each
        token in a way that they can be directly interpreted as
        probabilities for the next token. This means there is no
        need for a weighted sum of previous activations.
        Each item in this array represents the probabilities
        for the next item
        """

        B, T = idx.shape
        # C is embedding dimention

        token_embeddings = self.token_embedding_table(idx)  # (B, T, C)

        pos_embeddings = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)

        x = token_embeddings + pos_embeddings  # (B, T, C)

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
            # get predictions
            logits, loss = self(idx)

            # logit have dimention of (B, T, C)
            # focus only on the last time step
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

optimiser = torch.optim.AdamW(m.parameters(), lr=1e-3)


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

print(loss.item())

# generate output from model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
