import time
import torch
from model import GPT
from data_utils import create_vocab, encode, decode

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('data/names.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# uncomment for tinyshakespeare
# with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
#     text = f.read()

chars, stoi, itos = create_vocab(text)
vocab_size = len(chars)

model = GPT(vocab_size=vocab_size).to(device)

model.load_state_dict(torch.load("models/GPTv2.0.pt", map_location=device))
# model.load_state_dict(torch.load("models/tinyshakespeare.pt", map_location=device)) # uncomment for tinyshakespeare
model.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
output = model.generate(context, max_new_tokens=2000)

# literally just to be pretentious. 
res = decode(output[0].tolist(), itos)
for char in res: 
    print(char, end='', flush=True)
    time.sleep(0.02)