"""
Avacchedaka (Limitor) Attention.

Each attention edge (i, j) is factored as a typed triple:
    (r, p, q)  with  r in R  (relation),
                     p in K  (source aspect of token i),
                     q in K  (target aspect of token j).

This recovers the Navya-Nyaya structure in which every cognitive
relation is specified by limitors (avacchedaka) indicating *which*
aspect participates *under which* relation.

Standard multi-head attention is the K=1 special case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AvaConfig:
    d_model: int
    n_aspects: int          # K
    n_relations: int        # R
    d_aspect: int | None = None     # defaults to d_model // K
    d_relation: int | None = None   # defaults to d_aspect
    dropout: float = 0.0
    edge_temperature: float = 1.0
    softmax_mode: str = "per_q"  # "joint" (over j,q together) or "per_q" (over j for each q)
    causal: bool = False    # if True, position i attends only to j <= i
    topk_q: int | None = None  # if set, sparse top-k limitor gating: keep only top-k q-aspects per (i,r,p)

    def resolved(self) -> "AvaConfig":
        d_aspect = self.d_aspect or self.d_model // self.n_aspects
        d_relation = self.d_relation or d_aspect
        assert self.d_model == self.n_aspects * d_aspect, (
            f"d_model={self.d_model} must equal n_aspects*d_aspect="
            f"{self.n_aspects}*{d_aspect}"
        )
        return AvaConfig(
            d_model=self.d_model,
            n_aspects=self.n_aspects,
            n_relations=self.n_relations,
            d_aspect=d_aspect,
            d_relation=d_relation,
            dropout=self.dropout,
            edge_temperature=self.edge_temperature,
            softmax_mode=self.softmax_mode,
            causal=self.causal,
            topk_q=self.topk_q,
        )


class AvacchedakaAttention(nn.Module):
    """Limitor-indexed attention.

    For each token i we compute, for every aspect p and relation r,
    an attention distribution over (j, q) pairs: "for aspect p of token i
    under relation r, which aspect q of which token j is the relevant
    limitor?"
    """

    def __init__(self, cfg: AvaConfig):
        super().__init__()
        cfg = cfg.resolved()
        self.cfg = cfg
        R, K = cfg.n_relations, cfg.n_aspects
        d_a, d_r = cfg.d_aspect, cfg.d_relation

        # Typed projections. Shape: [R, K, d_aspect, d_relation].
        # W_Q[r, p] projects aspect p of a token into the query subspace of relation r.
        # W_K[r, q], W_V[r, q] project aspect q into key/value subspaces of relation r.
        # Init matches PyTorch's default nn.Linear (kaiming_uniform with a=sqrt(5)):
        # bound = sqrt(1 / (3 * fan_in)). This is crucial for parity with MHA baselines
        # built from nn.Linear modules -- naive uniform(-1/sqrt(fan_in), ...) is 1.73x
        # too large and changes learning dynamics substantially.
        def _init(*shape):
            t = torch.empty(*shape)
            fan_in = shape[-2]
            bound = 1.0 / math.sqrt(fan_in)  # matches nn.Linear default (kaiming_uniform a=sqrt(5)).
            nn.init.uniform_(t, -bound, bound)
            return nn.Parameter(t)

        # Q/K/V projections see the FULL token (dim d_model), not just one
        # aspect subspace. This is faithful to Navya-Nyaya: a limitor
        # (avacchedaka) is a *label* on the edge indicating which aspect
        # participates, not a capacity restriction on the cognition itself.
        # The typed structure lives in the output routing (W_O).
        d_model = cfg.d_model
        self.W_Q = _init(R, K, d_model, d_r)  # [R, K_src(=p), d_model, d_r]
        self.W_K = _init(R, K, d_model, d_r)  # [R, K_tgt(=q), d_model, d_r]
        self.W_V = _init(R, K, d_model, d_r)
        # W_O[r, p] projects relation-r messages back into aspect p of i.
        self.W_O = _init(R, K, d_r, d_a)

        # Optional aspect-mixer (see ablation in paper). When disabled, token
        # aspects are a fixed partition of the input vector.
        self.use_aspect_mixer = False
        if self.use_aspect_mixer:
            self.aspect_mixer = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            with torch.no_grad():
                self.aspect_mixer.weight.copy_(torch.eye(cfg.d_model))

        self.attn_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_edges: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, D] input.
            attn_mask: [N, N] additive mask (0 or -inf); broadcast across batch/rel.
            return_edges: if True also return the [B, R, N, K, N, K] edge tensor.
        Returns:
            [B, N, D] output; edge tensor if requested.
        """
        B, N, D = x.shape
        R = self.cfg.n_relations
        K = self.cfg.n_aspects
        d_a = self.cfg.d_aspect
        d_r = self.cfg.d_relation

        # Q/K/V projections see the FULL token x (dim d_model). The "aspect"
        # index on W_Q[r, p] indexes the *limitor label* on the edge, not a
        # restriction on which bits of x the query reads.
        x_full = self.aspect_mixer(x) if self.use_aspect_mixer else x  # [B, N, d_model]

        # Q[b, r, n, p, :] = x_full[b, n, :] @ W_Q[r, p, :, :]
        #   d indexes d_model (input), e indexes d_relation (output).
        Q = torch.einsum("bnd,rpde->brnpe", x_full, self.W_Q)   # [B, R, N, K_src, d_r]
        Kt = torch.einsum("bnd,rqde->brnqe", x_full, self.W_K)  # [B, R, N, K_tgt, d_r]
        Vt = torch.einsum("bnd,rqde->brnqe", x_full, self.W_V)

        # Scores: s[b, r, i, p, j, q] = Q[b,r,i,p,:] . K[b,r,j,q,:] / sqrt(d_r)
        scale = 1.0 / (math.sqrt(d_r) * self.cfg.edge_temperature)
        scores = torch.einsum("brnpe,brmqe->brnpmq", Q, Kt) * scale

        # Build the additive mask. Two sources may add together:
        #   (a) caller-provided attn_mask (e.g. padding mask)
        #   (b) causal mask if cfg.causal is True
        # Shape we'll add: [1, 1, N_q, 1, N_k, 1] (broadcasts over B, R, K_src, K_tgt).
        full_mask = None
        if attn_mask is not None:
            full_mask = attn_mask  # [N, N]
        if self.cfg.causal:
            i_idx = torch.arange(N, device=x.device)
            j_idx = torch.arange(N, device=x.device)
            causal = (j_idx[None, :] > i_idx[:, None]).to(scores.dtype) * float("-inf")
            # Convert -inf*0 nan to 0 for non-blocked positions.
            causal = torch.where(torch.isnan(causal), torch.zeros_like(causal), causal)
            full_mask = causal if full_mask is None else (full_mask + causal)
        if full_mask is not None:
            scores = scores + full_mask[None, None, :, None, :, None]

        # Optional sparse top-k limitor gating: for each (b, r, i, p) keep only
        # the top-k q values (target aspects) and mask the rest. Reduces compute
        # of the V aggregate but preserves correctness for k = K.
        if self.cfg.topk_q is not None and self.cfg.topk_q < K:
            # scores shape: [B, R, N, K, N, K]. Score per q (collapsing j) by max.
            q_score = scores.max(dim=-2).values   # [B, R, N, K, K]
            topk_vals, topk_idx = q_score.topk(self.cfg.topk_q, dim=-1)
            keep = torch.zeros_like(q_score, dtype=torch.bool)
            keep.scatter_(-1, topk_idx, True)     # [B, R, N, K, K] mask
            keep = keep[:, :, :, :, None, :].expand(-1, -1, -1, -1, N, -1)
            scores = torch.where(keep, scores, torch.full_like(scores, float("-inf")))

        if self.cfg.softmax_mode == "joint":
            # Softmax jointly over (j, q) for fixed (b, r, i, p).
            s_flat = scores.reshape(B, R, N, K, N * K)
            alpha_flat = F.softmax(s_flat, dim=-1)
            alpha_flat = self.attn_dropout(alpha_flat)
            alpha = alpha_flat.view(B, R, N, K, N, K)
        elif self.cfg.softmax_mode == "per_q":
            # Softmax over j only, for each (b, r, i, p, q). Stronger gradient
            # signal per typed edge; aspects q contribute additively to output.
            # scores shape: [B, R, N, K, N, K] -> softmax over dim=-2 (the N-for-j axis).
            alpha = F.softmax(scores, dim=-2)
            # If a softmax row was all -inf (e.g. top-k masked out an entire q,
            # or causal+padding masked everything), softmax returns NaN. Set to
            # 0 so that channel contributes nothing to the V aggregate.
            alpha = torch.nan_to_num(alpha, nan=0.0)
            alpha = self.attn_dropout(alpha)
        else:
            raise ValueError(self.cfg.softmax_mode)

        # Aggregate values: weighted_V[b,r,i,p,:] = sum_{j,q} alpha[...] * V[b,r,j,q,:]
        weighted_V = torch.einsum("brnpmq,brmqe->brnpe", alpha, Vt)  # [B,R,N,K,d_r]

        # Project each relation-r message back into aspect p of token i, then sum over r.
        out_asp = torch.einsum("brnpe,rped->bnpd", weighted_V, self.W_O)  # [B,N,K,d_a]

        out = out_asp.reshape(B, N, D)
        # Cache the most recent attention tensor so external code (e.g. the
        # interpretability hook) can read it without re-running forward().
        self.last_alpha = alpha.detach()
        if return_edges:
            return out, alpha
        return out

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
