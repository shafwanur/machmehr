import torch
from torch.utils.tensorboard import SummaryWriter
from model import GPT
from data_utils import create_vocab, encode, decode, get_batch

# Hyperparameters
batch_size = 64
block_size = 30 # should match block_size in model.py
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

# Load data
with open('data/names.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars, stoi, itos = create_vocab(text)
vocab_size = len(chars)

data = torch.tensor(encode(text, stoi), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

model = GPT(vocab_size=vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
writer = SummaryWriter(log_dir="runs/gpt_stats")

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split, split_data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split_data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        writer.add_scalar("loss/train", losses['train'], iter)
        writer.add_scalar("loss/val", losses['val'], iter)

    xb, yb = get_batch(train_data, block_size, batch_size, device)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "models/GPTv2.0.pt")
writer.close()
