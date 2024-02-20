import os
import numpy as np
import torch
from model import GPT
import pickle
import wandb
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

learning_rate = 3e-4
n_epoch = 3000
eval_interval = 500
eval_iter = 200
block_size = 256
batch_size = 8
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
grad_clip = 1.0
best_val_loss = float('inf')
best_model = None
bg_iter = 0
bias = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config_keys = [k for k, v in locals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}

parser = argparse.ArgumentParser()
parser.add_argument('-m', type=int, default=1)
parser.add_argument('-gpt', type=str, default='gpt2')
parser.add_argument('--wandb', action='store_true')
args = parser.parse_args()
init_from_mode = args.m
wandb_log = args.wandb

if init_from_mode == 1:
    init_from = 'scratch'
elif init_from_mode == 2:
    init_from = 'resume'
elif init_from_mode == 3:
    init_from = args.gpt
else:
    raise ValueError("Unknown init_from_mode value")

if wandb_log:
    wandb.init(project='nanoGPT', config=config)

with open('meta.pkl', 'rb') as file:
    meta = pickle.load(file)
vocab_size = meta['vocab_size']
train_data = np.memmap('train_data.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('val_data.bin', dtype=np.uint16, mode='r')

model_args = dict(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    bias=bias
)
ckpt = None

if init_from == 'scratch':
    model = GPT(vocab_size, block_size, n_layer, n_head, n_embd, dropout)
elif init_from == 'resume':
    ckpt = torch.load('best_model.pth', map_location=device)
    args = ckpt['model_args']
    model_args = {k: v for k, v in args.items() if k in ['vocab_size','block_size','n_layer', 'n_head', 'n_embd', 'dropout', 'bias' ]}
    model = GPT(**model_args)
    state_dict = ckpt['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    best_val_loss = ckpt['best_val_loss']
    bg_iter = ckpt['bg_iter']

elif init_from in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
    model = GPT.from_pretrained(init_from)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

else:
    raise ValueError("Unknown init_from value")

model.to(device)

optimizer = model.configure_optimizers(learning_rate, weight_decay, (beta1, beta2))
if init_from == 'resume':
    optimizer.load_state_dict(ckpt['optimizer'])
# 混合精度训练，加速训练，梯度缩放是通过乘以一个缩放因子（通常称为 GradScaler）来实现的，该因子用于将梯度值放大到适当的范围。这样做是为了避免梯度在较低精度下溢出并导致数值不稳定。
# 然而，在应用梯度缩放后，为了进行梯度裁剪（torch.nn.utils.clip_grad_norm_），需要将梯度值还原到其未缩放的范围，以便正确进行裁剪。
scaler = torch.cuda.amp.GradScaler()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epoch, eta_min=0)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


def get_batch(split):
    if split == 'train':
        data = train_data
    else:
        data = val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + block_size + 1].astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin_memory is faster than to(device), non_blocking can be set to True for faster data transfer
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def evaluate():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        out[split] = []
        for _ in range(eval_iter):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            out[split].append(loss.item())
        out[split] = np.mean(out[split])
    model.train()
    return out


for i in range(bg_iter, n_epoch):
    model.train()
    x, y = get_batch('train')
    logits, loss = model(x, y)
    scaler.scale(loss).backward()
    # 要先缩放回来，再进行梯度裁剪
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # 梯度裁剪, 防止梯度爆炸
    scaler.step(optimizer)
    scheduler.step()
    scaler.update()
    optimizer.zero_grad()

    if i % eval_interval == 0:
        out = evaluate()
        print(f"train loss: {out['train']} val loss: {out['val']}")
        if wandb_log:
            wandb.log({
                "iter": i,
                "train/loss": out['train'],
                "val/loss": out['val'],
                "lr": learning_rate
            })
        if out['val'] < best_val_loss:
            best_val_loss = out['val']
            best_model = model
            ckpt = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'best_val_loss': best_val_loss,
                'bg_iter': i
            }
            torch.save(ckpt, 'best_model.pth')






