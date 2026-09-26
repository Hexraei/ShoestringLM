"""The model: a decoder-only transformer, Llama-flavoured but written to be read.

Design choices (all standard, all credited in the README):
- RMSNorm instead of LayerNorm (cheaper, used by Llama).
- Rotary position embeddings (RoPE) instead of learned positions.
- Tied input/output embeddings: the output head IS the embedding matrix
  transposed, so the biggest single matrix is only counted once.
- Plain GELU MLP. No tricks. Tricks are for people with a GPU budget.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def _rope_tables(seq_len: int, head_dim: int, theta: float, device):
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    outer = torch.outer(t, freqs)  # (seq, half)
    return outer.cos(), outer.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (batch, heads, seq, head_dim)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope_theta = cfg.rope_theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = _rope_tables(t, self.head_dim, self.rope_theta, x.device)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.o(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ShoestringLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Tied head: logits come from the embedding matrix itself.
        logits = x @ self.embed.weight.T
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_parameters(self) -> dict:
        """Breakdown by part. With tied embeddings the head is not separate."""
        emb = self.embed.weight.numel()
        per_block = sum(p.numel() for p in self.blocks[0].parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            "embeddings (tied with output head)": emb,
            "per transformer block": per_block,
            f"all {self.cfg.n_layers} blocks": per_block * self.cfg.n_layers,
            "final norm": sum(p.numel() for p in self.norm.parameters()),
            "TOTAL trainable parameters": total,
        }

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new: int = 80, temperature: float = 0.9, top_k: int = 50):
        for _ in range(max_new):
            window = ids[:, -self.cfg.max_seq_len :]
            logits, _ = self(window)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                cut = logits.topk(top_k).values[:, -1, None]
                logits = logits.masked_fill(logits < cut, -float("inf"))
            probs = F.softmax(logits, dim=-1)
            ids = torch.cat([ids, torch.multinomial(probs, 1)], dim=1)
        return ids
