import torch
import torch.nn as nn
import math

# -----------------------
# Layer Norm
# -----------------------
class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

# -----------------------
# Feed Forward
# -----------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------
# Relative Positional Encoding
# -----------------------
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.d_model = d_model

        inv_freq = 1 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        pos_seq = torch.arange(seq_len - 1, -1.0, -1.0, device=self.inv_freq.device)
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        return pos_emb

# -----------------------
# MultiHead Attention (Transformer-XL style)
# -----------------------
class MultiHeadAttentionXL(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)

        self.r = nn.Linear(d_model, d_model, bias=False)

        self.o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.u = nn.Parameter(torch.Tensor(n_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(n_heads, self.d_head))

        nn.init.normal_(self.u, 0.0, 0.02)
        nn.init.normal_(self.v_bias, 0.0, 0.02) 

    def relative_shift(self, x):
        B, H, T, S = x.shape
        zero_pad = torch.zeros((B, H, T, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(B, H, S + 1, T)
        x_shifted = x_padded[:, :, 1:].view_as(x)
        return x_shifted

    def forward(self, x, mem, r_emb, mask=None):
        if mem is not None:
            mem = mem.detach()
            x_cat = torch.cat([mem, x], dim=1)
        else:
            x_cat = x

        q = self.q(x)
        k = self.k(x_cat)
        v = self.v(x_cat)
        r = self.r(r_emb)

        B, T, _ = q.shape
        _, S, _ = k.shape

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        r = r.view(r.shape[0], self.n_heads, self.d_head)

        AC = torch.einsum('bhid,bhjd->bhij', q + self.u.unsqueeze(1), k)

        BD = torch.einsum('bhid,jhd->bhij', q + self.v_bias.unsqueeze(1), r[-S:])
        BD = self.relative_shift(BD)

        attn = (AC + BD) / math.sqrt(self.d_head)

        mask = torch.tril(torch.ones(T, S, device=x.device)).bool()
        if S > T:
            mask = torch.cat([torch.ones(T, S - T, device=x.device, dtype=torch.bool), mask], dim=1)
        attn = attn.masked_fill(~mask, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o(out)

# -----------------------
# Transformer-XL Block
# -----------------------
class TransformerXLBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttentionXL(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mem, r_emb, mask):
        x = x + self.dropout(self.attn(self.norm1(x), mem, r_emb, mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

# -----------------------
# Transformer-XL Model
# -----------------------
class TransformerXL(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4, d_ff=256, mem_len=100, dropout=0.1):
        super().__init__()

        self.mem_len = mem_len
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = RelativePositionalEncoding(d_model)

        self.layers = nn.ModuleList([
            TransformerXLBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = LayerNormalization(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def init_mems(self, batch_size):
        return [None for _ in range(len(self.layers))]

    def update_mems(self, hiddens, mems):
        new_mems = []
        for h, m in zip(hiddens, mems):
            if m is None:
                new_m = h[:, -self.mem_len:]
            else:
                new_m = torch.cat([m, h], dim=1)[:, -self.mem_len:]
            new_mems.append(new_m.detach())
        return new_mems

    def forward(self, x, mems=None):
        B, T = x.shape

        if mems is None:
            mems = self.init_mems(B)

        x = self.embedding(x) * math.sqrt(self.d_model)

        r_emb = self.pos_emb(T + self.mem_len)

        hiddens = []

        for i, layer in enumerate(self.layers):
            mem = mems[i]
            x = layer(x, mem, r_emb, mask=None)
            hiddens.append(x)

        x = self.norm(x)
        logits = self.proj(x)

        new_mems = self.update_mems(hiddens, mems)

        return logits, new_mems

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    model = TransformerXL(vocab_size=10000)
    x = torch.randint(0, 10000, (2, 32))

    mems = model.init_mems(batch_size=2)

    logits, mems = model(x, mems)
    print(logits.shape)  # (batch, seq_len, vocab_size)
