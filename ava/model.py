"""Minimal transformer parametrized by attention kind (AvA or MHA baseline)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AvaConfig, AvacchedakaAttention


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention, used as the param-matched baseline.

    With n_heads = n_relations and head_dim = d_relation, its parameter count
    matches AvA when d_model == n_aspects * d_aspect and n_aspects == n_heads.
    Supports causal masking via the `causal` flag.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 causal: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        H, dh = self.n_heads, self.head_dim

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, N, H, dh).transpose(1, 2)  # [B, H, N, dh]

        q = split(self.W_Q(x))
        k = split(self.W_K(x))
        v = split(self.W_V(x))

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)  # [B, H, N, N]
        if self.causal:
            mask = torch.triu(torch.full((N, N), float("-inf"), device=x.device), diagonal=1)
            scores = scores + mask[None, None]
        if attn_mask is not None:
            scores = scores + attn_mask[None, None]
        alpha = self.drop(F.softmax(scores, dim=-1))
        out = alpha @ v  # [B, H, N, dh]
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.W_O(out)


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4            # for MHA baseline
    n_aspects: int = 4          # for AvA (K)
    n_relations: int = 4        # for AvA (R)
    d_relation: int | None = None
    d_aspect: int | None = None
    max_seq_len: int = 64
    dropout: float = 0.0
    attention: str = "ava"      # "ava" or "mha"
    causal: bool = False        # autoregressive masking
    topk_q: int | None = None   # AvA top-k limitor gating


class Block(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        if cfg.attention == "ava":
            ava_cfg = AvaConfig(
                d_model=cfg.d_model,
                n_aspects=cfg.n_aspects,
                n_relations=cfg.n_relations,
                d_aspect=cfg.d_aspect,
                d_relation=cfg.d_relation,
                dropout=cfg.dropout,
                causal=cfg.causal,
                topk_q=cfg.topk_q,
            )
            self.attn = AvacchedakaAttention(ava_cfg)
            self._attn_kind = "ava"
        elif cfg.attention == "mha":
            self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout, causal=cfg.causal)
            self._attn_kind = "mha"
        else:
            raise ValueError(cfg.attention)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    """Minimal transformer encoder -> token-level logits.

    Bidirectional (no causal mask) — appropriate for the binding task where
    the answer is read off a designated position and depends on all context.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # weight tying
        self.head.weight = self.tok.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, N = idx.shape
        pos = torch.arange(N, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
