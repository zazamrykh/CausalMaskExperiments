import math

import torch
import torch.nn as nn

from datasets import pad_token_idx


class RNNModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_dim=512,
                 rnn_cell="LSTM",
                 padding_idx=pad_token_idx,
                 num_layers=3):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_dim,
                                       padding_idx=padding_idx)
        self.ln = nn.LayerNorm(hidden_dim)
        self.rnn = getattr(nn, rnn_cell)(hidden_dim, hidden_dim,
                                         batch_first=True, dropout=0.3,
                                         num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.Linear(2 * hidden_dim, vocab_size)
        )

    def forward(self, x):
        x = self.embeddings(x)
        x = self.ln(x)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x


#  ------------------------------- Transformer architecture -------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_causal_mask=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."
        self.use_causal_mask = use_causal_mask

        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        max_len = 5000
        mask = torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.to_qkv(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.use_causal_mask:
            mask = self.mask[:,:,:T,:T]  # (1, 1, T, T)
            att = att.masked_fill(mask == 0, float('-inf'))

        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1,2).contiguous().reshape(B, T, C)
        out = self.out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1, use_causal_mask=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout, use_causal_mask)
        self.ff = FeedForward(embed_dim, ff_hidden_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_hidden_dim, max_len=5000, dropout=0.1, use_causal_mask=True):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = PositionalEncoding(embed_dim, max_len)
        self.use_causal_mask = use_causal_mask
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim, dropout, use_causal_mask) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.token_embed(x)
        x = self.pos_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
    