import torch
import pickle


temperature = 0.7
top_k = 5
max_token = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load('best_model.pth')
model.eval()
model.to(device)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


with open('meta.pkl', 'rb') as file:
    meta = pickle.load(file)
vocab_size = meta['vocab_size']
itos = meta['itos']
stoi = meta['stoi']

encode = lambda x: torch.tensor([stoi[ch] for ch in x], dtype=torch.long).to(device)
decode = lambda x: ''.join([itos[i] for i in x])

start = "The "
x = encode(start)

with torch.no_grad():
    y = model.generate(x, max_token, temperature, top_k)
    out = decode(y[0].tolist())
    print(out)


