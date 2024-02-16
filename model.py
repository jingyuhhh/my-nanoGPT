import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, n_embd, n_head, block_size,dropout=0.1, bias=False):
        super().__init__()
        head_size = n_embd // n_head
        self.query = nn.Linear(n_embd, head_size, bias)
        self.key = nn.Linear(n_embd, head_size, bias)
        self.value = nn.Linear(n_embd, head_size, bias)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        w = k @ q.transpose(-2, -1) / (C ** 0.5)
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        a = w @ v
        return a


class Multihead(nn.Module):
    def __init__(self, n_embd, n_head,block_size, dropout=0.1, bias=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, n_head,block_size, dropout, bias) for _ in range(n_head)])

    def forward(self, x):
        a = torch.cat([h(x) for h in self.heads], dim=-1)
        return a


class Block(nn.Module):
    def __init__(self, n_head, n_embd,block_size, dropout=0.1, bias=False):
        super().__init__()
        self.atten = Multihead(n_embd, n_head,block_size, dropout, bias)
        self.ln_1 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.atten(x)
        x = self.ln_1(x)
        x = x + self.mlp(x)
        x = self.ln_2(x)
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout=0.1, bias=False):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_head, n_embd, block_size,dropout, bias) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias)
        self.block_size = block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, y=None):
        pos_emb = self.pos_emb(torch.arange(self.block_size, device=x.device))

        x = self.tok_emb(x) + pos_emb
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        x = self.lm_head(x) # (B, T, vocab_size)
        if y is not None:
            B, T = y.shape
            x = x.view(B * T, -1)
            y = y.view(B * T)
            loss = F.cross_entropy(x, y)
        else:
            loss = None
        return x, loss

    def generate(self, x, max_token, temperature=1.0, top_k=None):
        for _ in range(max_token):
            x = x[-self.block_size:]
            logits = self.forward(x)
            logits = logits[:,-1,:] / temperature
            if top_k is not None:
                logits, _ = logits.topk(top_k, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            # multinomial找到概率最大的位置
            x = torch.cat([x, torch.multinomial(probs, 1)], dim=-1)
            print(torch.multinomial(probs, 1))
        return x

    def configure_optimizers(self, learning_rate, weight_decay=0.1, betas=(0.9, 0.95)):
        params = {pn: pv for pn, pv in self.named_parameters() if pv.requires_grad}
        # 让二维的才weight_decay
        decay_params = [pv for pn, pv in params.items() if pv.ndim >= 2]
        other_params = [pv for pn, pv in params.items() if pv.ndim < 2]
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": other_params, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=betas,
        )
        return optimizer

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        config =  {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }
        model_args = config[model_type]
        model_args['vocab_size'] = 50257
        model_args['block_size'] = 1024
        model_args['bias'] = False
        model_args['dropout'] = 0.1
        model = GPT(**model_args)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not(k.endswith('.attn.masked_bias') and k.endswith('.attn.bias'))]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


