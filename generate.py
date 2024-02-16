import torch
import pickle
from model import GPT


temperature = 0.8
top_k = 200
max_token = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ckpt = torch.load('best_model.pth')
model = GPT(**ckpt['model_args'])
model.load_state_dict(ckpt['model'])
model.to(device)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


with open('meta.pkl', 'rb') as file:
    meta = pickle.load(file)
itos = meta['itos']
stoi = meta['stoi']

encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[i] for i in x])

start = "First Citizen: "
x = encode(start)
x = torch.tensor(x, dtype=torch.long, device=device).view(1, -1) # (B, T)

with torch.no_grad():
    y = model.generate(x, max_token, temperature, top_k)
    out = decode(y[0].tolist())
    print(out)


