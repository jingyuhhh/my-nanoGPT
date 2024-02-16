import torch
import numpy as np
import pickle


with open('input.txt', 'r') as file:
    data = file.read()


chars = list(set(data))
vocab_size = len(chars)
data_size = len(data)

train_data = data[:int(data_size * 0.9)]
val_data = data[int(data_size * 0.9):]

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda x: torch.tensor([stoi[ch] for ch in x], dtype=torch.long)
train_data = encode(train_data)
val_data = encode(val_data)

train_data = np.array(train_data, dtype=np.uint16)
val_data = np.array(val_data, dtype=np.uint16)

train_data.tofile('train_data.bin')
val_data.tofile('val_data.bin')

meta = {
    'vocab_size': vocab_size,
    'stoi': stoi,
    'itos': itos

}

with open('meta.pkl', 'wb') as file:
    pickle.dump(meta, file)

