"""Shared readout head (STHead) and the causal Mamba context head of sequence tasks."""

import torch.nn as nn

HIDDEN = 256


class STHead(nn.Module):
    def __init__(self, n_tokens, d_model, n_outputs, dropout=0.1,
                 budget=19200, d_hidden=HIDDEN):
        super().__init__()
        d_token = max(1, round(budget / n_tokens))
        if d_token > d_model:
            d_token = d_model
        self.n_tokens, self.d_model, self.d_token = n_tokens, d_model, d_token
        self.flat_dim = n_tokens * d_token

        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_token),
        )
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.flat_dim, d_hidden), nn.LeakyReLU(0.01), nn.Dropout(dropout),
            nn.Linear(d_hidden, n_outputs),
        )
        self.n_outputs = n_outputs

    def extra_repr(self):
        return (f"n_tokens={self.n_tokens}, d_model={self.d_model}, "
                f"d_token={self.d_token}, flat_dim={self.flat_dim}, "
                f"n_outputs={self.n_outputs}")

    def readout(self, tokens):
        b, n, d = tokens.shape
        assert n == self.n_tokens and d == self.d_model, \
            f"STHead expects (B,{self.n_tokens},{self.d_model}), got {tokens.shape}"
        h = self.proj(tokens).flatten(1)
        for m in self.mlp[:-1]:
            h = m(h)
        return h

    def forward(self, tokens):
        return self.mlp[-1](self.readout(tokens))


class StreamContextHead(nn.Module):
    def __init__(self, n_outputs, context_len, d_model=HIDDEN, n_layers=2,
                 expand=4, d_state=16, d_conv=4, dropout=0.1):
        super().__init__()
        from mamba_ssm import Mamba
        self.context_len = context_len
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)])
        self.lns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, n_outputs)

    def extra_repr(self):
        return f"context_len={self.context_len}, n_outputs={self.out.out_features}, causal=True"

    def forward(self, h, valid_mask=None):
        b, l, d = h.shape
        assert l == self.context_len, \
            f"StreamContextHead expects L={self.context_len}, got {l}"
        x = h if valid_mask is None else h * valid_mask.unsqueeze(-1).to(h.dtype)
        for layer, ln in zip(self.layers, self.lns):
            x = x + self.drop(layer(ln(x)))
        return self.out(self.norm(x))
