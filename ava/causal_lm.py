"""Decoder-only causal language model using AvA or MHA attention.

Used for SCAN seq2seq (input+output concatenated) and for TinyStories LM
training. Provides:
  - CausalLM: GPT-style decoder-only transformer.
  - greedy_decode: autoregressive generation up to a stop token.
  - mask_cross_entropy: loss-only-on-output-tokens helper.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AvaConfig, AvacchedakaAttention
from .model import MultiHeadSelfAttention


@dataclass
class CausalLMConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_aspects: int = 4
    n_relations: int = 4
    d_relation: int | None = None
    d_aspect: int | None = None
    max_seq_len: int = 128
    dropout: float = 0.0
    attention: str = "ava"
    topk_q: int | None = None


class CausalBlock(nn.Module):
    def __init__(self, cfg: CausalLMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        if cfg.attention == "ava":
            ava_cfg = AvaConfig(
                d_model=cfg.d_model, n_aspects=cfg.n_aspects,
                n_relations=cfg.n_relations, d_aspect=cfg.d_aspect,
                d_relation=cfg.d_relation, dropout=cfg.dropout,
                causal=True, topk_q=cfg.topk_q,
            )
            self.attn = AvacchedakaAttention(ava_cfg)
        elif cfg.attention == "mha":
            self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout, causal=True)
        else:
            raise ValueError(cfg.attention)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model), nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalLM(nn.Module):
    def __init__(self, cfg: CausalLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([CausalBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, N = idx.shape
        assert N <= self.cfg.max_seq_len, f"sequence length {N} > max_seq_len {self.cfg.max_seq_len}"
        pos = torch.arange(N, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def greedy_decode(self, prompt: torch.Tensor, max_new: int, stop_token: int | None = None) -> torch.Tensor:
        """Greedy autoregressive generation. prompt: [B, N0]; returns [B, N0+max_new] (or earlier stop)."""
        self.eval()
        ids = prompt
        finished = torch.zeros(prompt.size(0), dtype=torch.bool, device=prompt.device)
        for _ in range(max_new):
            if ids.size(1) >= self.cfg.max_seq_len:
                break
            logits = self.forward(ids)[:, -1]  # [B, V]
            next_tok = logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            if stop_token is not None:
                finished = finished | (next_tok.squeeze(-1) == stop_token)
            ids = torch.cat([ids, next_tok], dim=1)
            if finished.all():
                break
        return ids


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on tokens where mask is 1, ignored where mask is 0.

    Args:
        logits: [B, N, V]
        targets: [B, N] (long), with shifted target at each position
        mask: [B, N] (float or bool), 1 on positions where loss is counted
    """
    B, N, V = logits.shape
    flat_logits = logits.reshape(B * N, V)
    flat_targets = targets.reshape(B * N)
    flat_mask = mask.reshape(B * N).to(flat_logits.dtype)
    loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")  # [B*N]
    return (loss * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)
