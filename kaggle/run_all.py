"""Self-contained Avacchedaka Attention experiment runner for Kaggle T4x2.

Embeds the AvA module, MHA baseline, binding task (weak + strict OOD),
multi-hop binding, and a battery loop. Targets Kaggle's T4x2 accelerator
(2x Tesla T4, sm_75, 16 GB each). Runs on CPU for local testing.
All output goes to /kaggle/working/ on Kaggle.

Local smoke test:
    python kaggle/run_all.py --smoke
On Kaggle, this becomes the kernel main and writes /kaggle/working/results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _resolve_device(prefer: str = "cuda") -> str:
    """Pick a working device. Kaggle T4 with default PyTorch works
    out of the box; this also catches any rare allocation failure."""
    if prefer == "cpu" or not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda")
        return "cuda"
    except RuntimeError as e:
        print(f"CUDA unusable ({e}); falling back to CPU.")
        return "cpu"


# =============================================================================
# Avacchedaka attention (the core module)
# =============================================================================

@dataclass
class AvaConfig:
    d_model: int
    n_aspects: int          # K
    n_relations: int        # R
    d_aspect: int | None = None
    d_relation: int | None = None
    dropout: float = 0.0
    causal: bool = False
    topk_q: int | None = None

    def resolved(self):
        d_a = self.d_aspect or self.d_model // self.n_aspects
        d_r = self.d_relation or d_a
        assert self.d_model == self.n_aspects * d_a
        return AvaConfig(
            d_model=self.d_model, n_aspects=self.n_aspects, n_relations=self.n_relations,
            d_aspect=d_a, d_relation=d_r, dropout=self.dropout,
            causal=self.causal, topk_q=self.topk_q,
        )


class AvacchedakaAttention(nn.Module):
    def __init__(self, cfg: AvaConfig):
        super().__init__()
        cfg = cfg.resolved()
        self.cfg = cfg
        R, K = cfg.n_relations, cfg.n_aspects
        d_a, d_r = cfg.d_aspect, cfg.d_relation

        def _init(*shape):
            t = torch.empty(*shape)
            bound = 1.0 / math.sqrt(shape[-2])
            nn.init.uniform_(t, -bound, bound)
            return nn.Parameter(t)

        d = cfg.d_model
        self.W_Q = _init(R, K, d, d_r)
        self.W_K = _init(R, K, d, d_r)
        self.W_V = _init(R, K, d, d_r)
        self.W_O = _init(R, K, d_r, d_a)
        self.attn_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        R, K = self.cfg.n_relations, self.cfg.n_aspects
        d_a, d_r = self.cfg.d_aspect, self.cfg.d_relation
        Q = torch.einsum("bnd,rpde->brnpe", x, self.W_Q)
        Kt = torch.einsum("bnd,rqde->brnqe", x, self.W_K)
        Vt = torch.einsum("bnd,rqde->brnqe", x, self.W_V)
        scale = 1.0 / math.sqrt(d_r)
        scores = torch.einsum("brnpe,brmqe->brnpmq", Q, Kt) * scale
        if self.cfg.causal:
            i_idx = torch.arange(N, device=x.device)
            j_idx = torch.arange(N, device=x.device)
            causal = (j_idx[None, :] > i_idx[:, None]).to(scores.dtype) * float("-inf")
            causal = torch.where(torch.isnan(causal), torch.zeros_like(causal), causal)
            scores = scores + causal[None, None, :, None, :, None]
        if self.cfg.topk_q is not None and self.cfg.topk_q < K:
            q_score = scores.max(dim=-2).values  # [B, R, N, K, K]
            _, topk_idx = q_score.topk(self.cfg.topk_q, dim=-1)
            keep = torch.zeros_like(q_score, dtype=torch.bool)
            keep.scatter_(-1, topk_idx, True)
            keep = keep[:, :, :, :, None, :].expand(-1, -1, -1, -1, N, -1)
            scores = torch.where(keep, scores, torch.full_like(scores, float("-inf")))
        alpha = F.softmax(scores, dim=-2)  # softmax over j only.
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.attn_dropout(alpha)
        weighted_V = torch.einsum("brnpmq,brmqe->brnpe", alpha, Vt)
        out_asp = torch.einsum("brnpe,rped->bnpd", weighted_V, self.W_O)
        return out_asp.reshape(B, N, D)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 causal: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H, dh = self.n_heads, self.head_dim
        def split(t): return t.view(B, N, H, dh).transpose(1, 2)
        q, k, v = split(self.W_Q(x)), split(self.W_K(x)), split(self.W_V(x))
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        if self.causal:
            mask = torch.triu(torch.full((N, N), float("-inf"), device=x.device), diagonal=1)
            scores = scores + mask[None, None]
        alpha = self.drop(F.softmax(scores, dim=-1))
        out = (alpha @ v).transpose(1, 2).reshape(B, N, D)
        return self.W_O(out)


@dataclass
class TConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    n_aspects: int = 4
    n_relations: int = 4
    max_seq_len: int = 64
    dropout: float = 0.0
    attention: str = "ava"
    causal: bool = False
    topk_q: int | None = None


class Block(nn.Module):
    def __init__(self, cfg: TConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        if cfg.attention == "ava":
            self.attn = AvacchedakaAttention(AvaConfig(
                d_model=cfg.d_model, n_aspects=cfg.n_aspects,
                n_relations=cfg.n_relations, dropout=cfg.dropout,
                causal=cfg.causal, topk_q=cfg.topk_q))
        elif cfg.attention == "mha":
            self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout,
                                               causal=cfg.causal)
        else:
            raise ValueError(cfg.attention)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, cfg: TConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied embeddings

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, N = idx.shape
        pos = torch.arange(N, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Tasks
# =============================================================================

@dataclass
class BindingSpec:
    n_entities: int = 8
    n_properties: int = 4
    n_values: int = 16
    n_bindings: int = 4
    test_pair_frac: float = 0.25
    seed: int = 0


class BindingVocab:
    PAD, ANSWER = 0, 1
    def __init__(self, spec: BindingSpec):
        self.spec = spec
        self.entity_offset = 2
        self.property_offset = self.entity_offset + spec.n_entities
        self.value_offset = self.property_offset + spec.n_properties
        self.vocab_size = self.value_offset + spec.n_values
    def E(self, e): return self.entity_offset + e
    def P(self, p): return self.property_offset + p
    def V(self, v): return self.value_offset + v
    @property
    def value_range(self): return (self.value_offset, self.value_offset + self.spec.n_values)


def build_pair_split(spec: BindingSpec):
    rng = random.Random(spec.seed)
    pairs = [(e, p) for e in range(spec.n_entities) for p in range(spec.n_properties)]
    rng.shuffle(pairs)
    n_test = max(1, int(round(spec.test_pair_frac * len(pairs))))
    return pairs[n_test:], pairs[:n_test]


class BindingDataset(Dataset):
    def __init__(self, split, n_samples, spec, train_pairs, test_pairs, seed=0, strict=False):
        self.split, self.n_samples, self.spec = split, n_samples, spec
        self.vocab = BindingVocab(spec)
        self.train_pairs, self.test_pairs = train_pairs, test_pairs
        self.seed, self.strict = seed, strict

    def __len__(self): return self.n_samples

    def __getitem__(self, idx):
        rng = random.Random(f"{self.seed}-{self.split}-{idx}")
        pool = self.train_pairs if self.split == "train" else self.test_pairs
        e_q, p_q = rng.choice(pool)
        spec = self.spec
        if self.strict:
            allowed = self.train_pairs if self.split == "train" else self.test_pairs
        else:
            allowed = self.train_pairs + self.test_pairs
        other_p = [(e_q, p) for p in range(spec.n_properties) if p != p_q and (e_q, p) in allowed]
        other_e = [(e, p_q) for e in range(spec.n_entities) if e != e_q and (e, p_q) in allowed]
        chosen = [(e_q, p_q)]
        if other_p: chosen.append(rng.choice(other_p))
        if other_e: chosen.append(rng.choice(other_e))
        rest = [pr for pr in allowed if pr not in chosen]
        rng.shuffle(rest)
        while len(chosen) < spec.n_bindings and rest:
            chosen.append(rest.pop())
        chosen = chosen[: spec.n_bindings]
        rng.shuffle(chosen)
        values = [rng.randrange(spec.n_values) for _ in chosen]
        ans = values[chosen.index((e_q, p_q))]
        V = self.vocab
        toks = []
        for (e, p), v in zip(chosen, values):
            toks += [V.E(e), V.P(p), V.V(v)]
        toks += [V.E(e_q), V.P(p_q), V.ANSWER]
        return {
            "seq": torch.tensor(toks, dtype=torch.long),
            "target": torch.tensor(V.V(ans), dtype=torch.long),
            "answer_pos": torch.tensor(len(toks) - 1, dtype=torch.long),
        }


# Multi-hop binding -----------------------------------------------------------

@dataclass
class MultiHopSpec:
    n_entities: int = 8
    n_properties: int = 4
    n_values: int = 16
    n_bindings_A: int = 3
    n_bindings_B: int = 3
    seed: int = 0


class MultiHopVocab(BindingVocab):
    def __init__(self, spec: MultiHopSpec):
        super().__init__(BindingSpec(
            n_entities=spec.n_entities, n_properties=spec.n_properties,
            n_values=spec.n_values, seed=spec.seed))


def build_query_split(spec: MultiHopSpec, test_frac: float = 0.25):
    rng = random.Random(spec.seed)
    triples = [(e, p1, p2) for e in range(spec.n_entities)
               for p1 in range(spec.n_properties) for p2 in range(spec.n_properties) if p1 != p2]
    rng.shuffle(triples)
    n_test = max(1, int(round(test_frac * len(triples))))
    return triples[n_test:], triples[:n_test]


class MultiHopDataset(Dataset):
    def __init__(self, split, n_samples, spec, train_q, test_q, seed=0):
        self.split, self.n_samples, self.spec = split, n_samples, spec
        self.vocab = MultiHopVocab(spec)
        self.train_q, self.test_q, self.seed = train_q, test_q, seed

    def __len__(self): return self.n_samples

    def __getitem__(self, idx):
        rng = random.Random(f"{self.seed}-mh-{self.split}-{idx}")
        spec = self.spec
        pool = self.train_q if self.split == "train" else self.test_q
        e_q, p1, p2 = rng.choice(pool)
        e_mid_pool = [e for e in range(spec.n_entities) if e != e_q]
        e_mid = rng.choice(e_mid_pool)
        v_ans = rng.randrange(spec.n_values)
        type_A = [(e_q, p1, e_mid)]
        type_B = [(e_mid, p2, v_ans)]
        other_p = [p for p in range(spec.n_properties) if p != p1]
        if other_p:
            p_o = rng.choice(other_p)
            type_A.append((e_q, p_o, rng.choice(e_mid_pool)))
        e_other = rng.choice([e for e in range(spec.n_entities) if e not in (e_q, e_mid)])
        e_third = rng.choice([e for e in range(spec.n_entities) if e != e_other])
        type_A.append((e_other, p1, e_third))
        v_other = rng.randrange(spec.n_values)
        if v_other == v_ans: v_other = (v_other + 1) % spec.n_values
        type_B.append((e_other, p2, v_other))
        if other_p:
            type_B.append((e_mid, rng.choice(other_p), rng.randrange(spec.n_values)))
        type_A = type_A[: spec.n_bindings_A]
        type_B = type_B[: spec.n_bindings_B]
        all_b = [("A", *b) for b in type_A] + [("B", *b) for b in type_B]
        rng.shuffle(all_b)
        V = self.vocab
        toks = []
        for kind, a, b, c in all_b:
            if kind == "A":
                toks += [V.E(a), V.P(b), V.E(c)]
            else:
                toks += [V.E(a), V.P(b), V.V(c)]
        toks += [V.E(e_q), V.P(p1), V.P(p2), V.ANSWER]
        return {
            "seq": torch.tensor(toks, dtype=torch.long),
            "target": torch.tensor(V.V(v_ans), dtype=torch.long),
            "answer_pos": torch.tensor(len(toks) - 1, dtype=torch.long),
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# =============================================================================
# Training & evaluation
# =============================================================================

def evaluate(model, loader, device, value_range):
    model.eval()
    correct, total = 0, 0
    lo, hi = value_range
    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            target = batch["target"].to(device)
            pos = batch["answer_pos"].to(device)
            logits = model(seq)
            idx = torch.arange(logits.size(0), device=device)
            ans = logits[idx, pos]
            mask = torch.full_like(ans, float("-inf")); mask[:, lo:hi] = 0.0
            ans = ans + mask
            pred = ans.argmax(dim=-1)
            correct += (pred == target).sum().item()
            total += target.numel()
    model.train()
    return correct / total


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def train_one(*, task: str, attention: str, seed: int, n_train: int, lr: float,
              d_model: int, n_layers: int, n_heads: int, n_aspects: int, n_relations: int,
              batch_size: int, eval_every: int, device: str, strict: bool = False,
              max_seq_len: int = 64) -> dict:
    """Train one configuration end to end. Returns a dict of metrics."""
    set_seed(seed)
    if task == "binding":
        spec = BindingSpec(seed=0)
        train_pairs, test_pairs = build_pair_split(spec)
        vocab = BindingVocab(spec)
        train_set = BindingDataset("train", n_train, spec, train_pairs, test_pairs, seed=seed, strict=strict)
        iid_set = BindingDataset("train", 1500, spec, train_pairs, test_pairs, seed=seed + 1000, strict=strict)
        ood_set = BindingDataset("test", 1500, spec, train_pairs, test_pairs, seed=seed + 2000, strict=strict)
        seq_len = 15
    elif task == "multihop":
        spec = MultiHopSpec(seed=0)
        train_q, test_q = build_query_split(spec)
        vocab = MultiHopVocab(spec)
        train_set = MultiHopDataset("train", n_train, spec, train_q, test_q, seed=seed)
        iid_set = MultiHopDataset("train", 1500, spec, train_q, test_q, seed=seed + 1000)
        ood_set = MultiHopDataset("test", 1500, spec, train_q, test_q, seed=seed + 2000)
        seq_len = 22
    else:
        raise ValueError(task)

    pin = (device == "cuda")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, collate_fn=collate, pin_memory=pin, num_workers=0)
    iid_loader = DataLoader(iid_set, batch_size=256, collate_fn=collate, pin_memory=pin, num_workers=0)
    ood_loader = DataLoader(ood_set, batch_size=256, collate_fn=collate, pin_memory=pin, num_workers=0)

    cfg = TConfig(
        vocab_size=vocab.vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, n_aspects=n_aspects, n_relations=n_relations,
        max_seq_len=max(seq_len, max_seq_len), attention=attention, dropout=0.0,
    )
    model = TinyTransformer(cfg).to(device)
    n_params = model.n_parameters()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = max(1, n_train // batch_size)
    warmup = max(1, int(0.05 * total_steps))

    def lr_lambda(step):
        if step < warmup: return step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    lo, hi = vocab.value_range

    t0 = time.time()
    step = 0
    for batch in train_loader:
        seq = batch["seq"].to(device, non_blocking=pin)
        target = batch["target"].to(device, non_blocking=pin)
        pos = batch["answer_pos"].to(device, non_blocking=pin)
        logits = model(seq)
        idx = torch.arange(logits.size(0), device=device)
        ans = logits[idx, pos]
        mask = torch.full_like(ans, float("-inf")); mask[:, lo:hi] = 0.0
        ans = ans + mask
        loss = F.cross_entropy(ans, target)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        step += 1

    iid_acc = evaluate(model, iid_loader, device, (lo, hi))
    ood_acc = evaluate(model, ood_loader, device, (lo, hi))
    return {
        "task": task, "attention": attention, "seed": seed, "strict": strict,
        "n_params": n_params, "n_train": n_train, "d_model": d_model,
        "n_layers": n_layers, "n_aspects": n_aspects, "n_relations": n_relations,
        "iid_acc": iid_acc, "ood_acc": ood_acc, "elapsed_s": time.time() - t0,
    }


# =============================================================================
# Battery
# =============================================================================

def run_battery(out_dir: Path, device: str, smoke: bool = False, mode: str = "full") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_filename = {
        "tuned": "results_tuned.jsonl",
        "multihop_solve": "results_multihop.jsonl",
        "multihop_deep": "results_multihop_deep.jsonl",
        "scan": "results_scan_addjump.jsonl",
        "full": "results.jsonl",
    }.get(mode, "results.jsonl")
    log_path = out_dir / log_filename

    if smoke:
        # Tiny test, ~10 seconds, verifies every code path.
        configs = [
            dict(task="binding", attention="ava", seed=0, n_train=2000, lr=3e-3,
                 d_model=64, n_layers=2, n_heads=4, n_aspects=4, n_relations=4,
                 batch_size=64, eval_every=50, device=device, strict=False),
            dict(task="binding", attention="mha", seed=0, n_train=2000, lr=3e-3,
                 d_model=64, n_layers=2, n_heads=4, n_aspects=4, n_relations=4,
                 batch_size=64, eval_every=50, device=device, strict=True),
            dict(task="multihop", attention="ava", seed=0, n_train=2000, lr=3e-3,
                 d_model=64, n_layers=2, n_heads=4, n_aspects=4, n_relations=4,
                 batch_size=64, eval_every=50, device=device, strict=False),
        ]
    elif mode == "tuned":
        # Earlier tuned battery, kept for reproducibility.
        configs = []
        d_model, n_layers = 512, 8
        d_wide = int(d_model * 1.4); d_wide -= d_wide % 4
        n_train = 300000
        for seed in range(3):
            for arch_kw in (
                dict(attention="ava", d_model=d_model, n_heads=4, n_aspects=4, n_relations=4),
                dict(attention="mha", d_model=d_model, n_heads=4, n_aspects=4, n_relations=4),
                dict(attention="mha", d_model=d_wide, n_heads=4, n_aspects=4, n_relations=4),
            ):
                configs.append(dict(task="binding", seed=seed, n_train=n_train, lr=1e-3,
                    n_layers=n_layers, batch_size=256, eval_every=2000,
                    device=device, strict=True, **arch_kw))
        d_model, n_layers = 256, 8
        d_wide = int(d_model * 1.4); d_wide -= d_wide % 4
        n_train = 300000
        for seed in range(3):
            for arch_kw in (
                dict(attention="ava", d_model=d_model, n_heads=4, n_aspects=4, n_relations=4),
                dict(attention="mha", d_model=d_model, n_heads=4, n_aspects=4, n_relations=4),
                dict(attention="mha", d_model=d_wide, n_heads=4, n_aspects=4, n_relations=4),
            ):
                configs.append(dict(task="multihop", seed=seed, n_train=n_train, lr=1e-3,
                    n_layers=n_layers, batch_size=256, eval_every=2000,
                    device=device, strict=False, **arch_kw))
    elif mode == "multihop_solve":
        # Push to solve multi-hop: deeper model + longer training.
        # d=256 L=10 with 500k samples, lr=1e-3, 3 seeds x 3 archs = 9 runs.
        configs = []
        d_model, n_layers = 256, 10
        d_wide = int(d_model * 1.4); d_wide -= d_wide % 4
        n_train = 500000
        for seed in range(3):
            for arch_kw in (
                dict(attention="ava", d_model=d_model, n_heads=4, n_aspects=4, n_relations=4),
                dict(attention="mha", d_model=d_model, n_heads=4, n_aspects=4, n_relations=4),
                dict(attention="mha", d_model=d_wide, n_heads=4, n_aspects=4, n_relations=4),
            ):
                configs.append(dict(task="multihop", seed=seed, n_train=n_train, lr=1e-3,
                    n_layers=n_layers, batch_size=256, eval_every=4000,
                    device=device, strict=False, **arch_kw))
    elif mode == "scan":
        # SCAN simple was already covered in v13. Re-run only the hard split:
        # add_prim_jump (Lake & Baroni 2018 Table 5). Train without "jump" in
        # composition; test on compositions of "jump". Famously hard for
        # vanilla transformers (often <10% sequence accuracy).
        # 3 archs x 3 seeds = 9 runs.
        configs = []
        for split in ("addjump",):
            for seed in range(3):
                for arch_kw in (
                    dict(attention="ava", d_model=128, n_heads=4, n_aspects=4, n_relations=4),
                    dict(attention="mha", d_model=128, n_heads=4, n_aspects=4, n_relations=4),
                    dict(attention="mha", d_model=176, n_heads=4, n_aspects=4, n_relations=4),
                ):
                    configs.append(dict(_kind="scan", split=split, seed=seed,
                                         n_epochs=15, lr=1e-3, n_layers=4,
                                         batch_size=128, dropout=0.1,
                                         device=device, **arch_kw))
    elif mode == "multihop_deep":
        # L=12 with 500k samples. Goal: cross 80% with AvA, confirm
        # that the architectural advantage scales with depth.
        configs = []
        d_model, n_layers = 256, 12
        d_wide = int(d_model * 1.4); d_wide -= d_wide % 4
        n_train = 500000
        for seed in range(3):
            for arch_kw in (
                dict(attention="ava", d_model=d_model, n_heads=4, n_aspects=4, n_relations=4),
                dict(attention="mha", d_model=d_model, n_heads=4, n_aspects=4, n_relations=4),
                dict(attention="mha", d_model=d_wide, n_heads=4, n_aspects=4, n_relations=4),
            ):
                configs.append(dict(task="multihop", seed=seed, n_train=n_train, lr=1e-3,
                    n_layers=n_layers, batch_size=256, eval_every=4000,
                    device=device, strict=False, **arch_kw))
    else:
        # Real Kaggle battery. Two model scales x three tasks x three configs x five seeds.
        configs = []
        for d_model, n_layers in [(256, 6), (512, 8)]:
            d_wide = int(d_model * 1.4)
            d_wide -= d_wide % 4
            n_train = 200000
            for seed in range(5):
                ava_kw = dict(attention="ava", d_model=d_model, n_heads=4, n_aspects=4, n_relations=4)
                mha_kw = dict(attention="mha", d_model=d_model, n_heads=4, n_aspects=4, n_relations=4)
                mha_wide_kw = dict(attention="mha", d_model=d_wide, n_heads=4, n_aspects=4, n_relations=4)
                for task, strict in [("binding", False), ("binding", True), ("multihop", False)]:
                    for arch_kw in (ava_kw, mha_kw, mha_wide_kw):
                        configs.append(dict(
                            task=task, seed=seed, n_train=n_train, lr=3e-3,
                            n_layers=n_layers, batch_size=256, eval_every=400,
                            device=device, strict=strict, **arch_kw,
                        ))

    print(f"Running {len(configs)} configurations on device={device}")
    if log_path.exists(): log_path.unlink()
    for i, cfg in enumerate(configs):
        is_scan = cfg.pop("_kind", None) == "scan"
        try:
            r = train_scan_one(**cfg) if is_scan else train_one(**cfg)
        except Exception as e:
            r = {"error": str(e), **{k: v for k, v in cfg.items() if k != "device"}}
            print(f"[{i+1}/{len(configs)}] ERROR: {e}")
        with open(log_path, "a") as f:
            f.write(json.dumps(r) + "\n")
        if "error" not in r:
            tag = r.get("split") or r.get("strict")
            print(f"[{i+1}/{len(configs)}] {r['task']:8s} {tag} {r['attention']:3s} "
                  f"d={r['d_model']:>4} L={r['n_layers']} s{r['seed']} "
                  f"params={r['n_params']:>9,} iid={r['iid_acc']:.3f} ood={r['ood_acc']:.3f} "
                  f"({r['elapsed_s']:.1f}s)")


# =============================================================================
# Causal LM (decoder-only transformer)  +  SCAN dataset and trainer
# =============================================================================

class CausalBlock(nn.Module):
    def __init__(self, cfg: TConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        if cfg.attention == "ava":
            self.attn = AvacchedakaAttention(AvaConfig(
                d_model=cfg.d_model, n_aspects=cfg.n_aspects,
                n_relations=cfg.n_relations, dropout=cfg.dropout,
                causal=True, topk_q=cfg.topk_q))
        elif cfg.attention == "mha":
            self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout,
                                               causal=True)
        else:
            raise ValueError(cfg.attention)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalLM(nn.Module):
    def __init__(self, cfg: TConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([CausalBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx):
        B, N = idx.shape
        assert N <= self.cfg.max_seq_len
        pos = torch.arange(N, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def greedy_decode(self, prompt, max_new, stop_token=None):
        self.eval()
        ids = prompt
        finished = torch.zeros(prompt.size(0), dtype=torch.bool, device=prompt.device)
        for _ in range(max_new):
            if ids.size(1) >= self.cfg.max_seq_len:
                break
            logits = self.forward(ids)[:, -1]
            nxt = logits.argmax(dim=-1, keepdim=True)
            if stop_token is not None:
                finished = finished | (nxt.squeeze(-1) == stop_token)
            ids = torch.cat([ids, nxt], dim=1)
            if finished.all():
                break
        return ids


def masked_cross_entropy(logits, targets, mask):
    B, N, V = logits.shape
    fl = logits.reshape(B * N, V); ft = targets.reshape(B * N)
    fm = mask.reshape(B * N).to(fl.dtype)
    loss = F.cross_entropy(fl, ft, reduction="none")
    return (loss * fm).sum() / fm.sum().clamp_min(1.0)


# ---- SCAN dataset (downloaded at runtime) -------------------------------

import urllib.request

SCAN_URLS = {
    "simple_train": "https://raw.githubusercontent.com/brendenlake/SCAN/master/simple_split/tasks_train_simple.txt",
    "simple_test":  "https://raw.githubusercontent.com/brendenlake/SCAN/master/simple_split/tasks_test_simple.txt",
    "mcd1_train":   "https://raw.githubusercontent.com/brendenlake/SCAN/master/MCD_splits/tasks_train_mcd1.txt",
    "mcd1_test":    "https://raw.githubusercontent.com/brendenlake/SCAN/master/MCD_splits/tasks_test_mcd1.txt",
    "addjump_train":"https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_train_addprim_jump.txt",
    "addjump_test": "https://raw.githubusercontent.com/brendenlake/SCAN/master/add_prim_split/tasks_test_addprim_jump.txt",
}

def _scan_download(url, dest):
    if dest.exists() and dest.stat().st_size > 0: return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {dest.name}")
    urllib.request.urlretrieve(url, dest)

def fetch_scan(split, cache_dir="/tmp/scan_data"):
    cache = Path(cache_dir)
    train_path = cache / f"{split}_train.txt"; test_path = cache / f"{split}_test.txt"
    _scan_download(SCAN_URLS[f"{split}_train"], train_path)
    _scan_download(SCAN_URLS[f"{split}_test"], test_path)
    def _parse(p):
        out = []
        with open(p) as f:
            for ln in f:
                ln = ln.strip()
                if "IN:" not in ln or "OUT:" not in ln: continue
                a, b = ln.split("OUT:")
                out.append((a.replace("IN:", "").strip(), b.strip()))
        return out
    return _parse(train_path), _parse(test_path)


class ScanVocab:
    PAD, SEP, EOS = 0, 1, 2
    def __init__(self, pairs):
        in_w, out_w = set(), set()
        for x, y in pairs:
            in_w.update(x.split()); out_w.update(y.split())
        self.input_tokens = sorted(in_w); self.output_tokens = sorted(out_w)
        self.tok2id = {"<pad>": 0, "<sep>": 1, "<eos>": 2}
        for w in self.input_tokens: self.tok2id[w] = len(self.tok2id)
        for w in self.output_tokens: self.tok2id[w] = len(self.tok2id)
        self.vocab_size = len(self.tok2id)
        self.id2tok = {v: k for k, v in self.tok2id.items()}

    def encode_pair(self, x, y, max_len):
        x_ids = [self.tok2id[w] for w in x.split()]
        y_ids = [self.tok2id[w] for w in y.split()]
        seq = x_ids + [self.SEP] + y_ids + [self.EOS]
        if len(seq) > max_len: seq = seq[:max_len]
        n_input = len(x_ids) + 1
        mask = [0] * len(seq)
        for i in range(n_input, len(seq)): mask[i] = 1
        return seq, mask, n_input


class ScanDataset(Dataset):
    def __init__(self, pairs, vocab, max_len):
        self.pairs = pairs; self.vocab = vocab; self.max_len = max_len
        self.encoded = [vocab.encode_pair(x, y, max_len) for x, y in pairs]

    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        seq, mask, n_in = self.encoded[i]
        return {"seq": torch.tensor(seq), "mask": torch.tensor(mask, dtype=torch.float),
                "n_input": torch.tensor(n_in)}


def scan_collate(batch, pad_id=0):
    M = max(b["seq"].size(0) for b in batch); B = len(batch)
    seq = torch.full((B, M), pad_id, dtype=torch.long)
    mask = torch.zeros(B, M, dtype=torch.float)
    nin = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["seq"].size(0)
        seq[i, :L] = b["seq"]; mask[i, :L] = b["mask"]; nin[i] = b["n_input"]
    return {"seq": seq, "mask": mask, "n_input": nin}


def scan_evaluate(model, vocab, pairs, max_len, device, max_new=None):
    if max_new is None: max_new = max_len - 1
    model.eval()
    n_correct, n_total = 0, 0
    with torch.no_grad():
        for x, y in pairs:
            x_ids = [vocab.tok2id[w] for w in x.split()] + [vocab.SEP]
            y_ids = [vocab.tok2id[w] for w in y.split()] + [vocab.EOS]
            prompt = torch.tensor([x_ids], dtype=torch.long, device=device)
            if prompt.size(1) >= max_len: continue
            out = model.greedy_decode(prompt, max_new=min(max_new, max_len - prompt.size(1)),
                                       stop_token=vocab.EOS)
            generated = out[0, prompt.size(1):].tolist()
            if vocab.EOS in generated:
                generated = generated[: generated.index(vocab.EOS) + 1]
            n_correct += int(generated == y_ids); n_total += 1
    model.train()
    return n_correct / max(1, n_total)


def train_scan_one(*, attention, split, seed, n_epochs, lr, d_model, n_layers,
                    n_heads, n_aspects, n_relations, batch_size, dropout, device,
                    max_seq_len=128):
    set_seed(seed)
    train_pairs, test_pairs = fetch_scan(split)
    vocab = ScanVocab(train_pairs + test_pairs)
    Lmax = 0
    for x, y in train_pairs + test_pairs:
        Lmax = max(Lmax, len(x.split()) + 1 + len(y.split()) + 1)
    Lmax = min(max_seq_len, Lmax)
    train_ds = ScanDataset(train_pairs, vocab, Lmax)
    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=scan_collate, pin_memory=pin, num_workers=0)
    cfg = TConfig(vocab_size=vocab.vocab_size, d_model=d_model, n_layers=n_layers,
                  n_heads=n_heads, n_aspects=n_aspects, n_relations=n_relations,
                  max_seq_len=Lmax, dropout=dropout, attention=attention)
    model = CausalLM(cfg).to(device)
    n_params = model.n_parameters()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = n_epochs * len(train_loader)
    warmup = max(1, int(0.05 * total_steps))
    def lr_lambda(s):
        if s < warmup: return s / warmup
        p = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    t0 = time.time()
    step = 0
    last_tok_acc = 0.0
    for epoch in range(n_epochs):
        for batch in train_loader:
            seq = batch["seq"].to(device, non_blocking=pin)
            mask = batch["mask"].to(device, non_blocking=pin)
            logits = model(seq[:, :-1])
            targets = seq[:, 1:]
            loss_mask = mask[:, 1:]
            loss = masked_cross_entropy(logits, targets, loss_mask)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
        # Per-epoch summary on train batch.
        with torch.no_grad():
            pred = logits.argmax(-1)
            last_tok_acc = (((pred == targets) * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)).item()
    seq_acc = scan_evaluate(model, vocab, test_pairs, Lmax, device, max_new=Lmax)
    return {
        "task": "scan", "split": split, "attention": attention, "seed": seed,
        "n_params": n_params, "d_model": d_model, "n_layers": n_layers,
        "n_aspects": n_aspects, "n_relations": n_relations, "n_epochs": n_epochs,
        "iid_acc": last_tok_acc, "ood_acc": seq_acc, "elapsed_s": time.time() - t0,
    }


def default_out_dir() -> str:
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working"
    return "./out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3-config smoke test")
    ap.add_argument("--mode", default="scan",
                    choices=["full", "tuned", "multihop_solve", "multihop_deep", "scan"],
                    help="full | tuned | multihop_solve | multihop_deep | scan")
    ap.add_argument("--out_dir", default=default_out_dir())
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"Ignoring unrecognized args from notebook wrapper: {unknown}")
    if args.device == "auto":
        device = _resolve_device("cuda")
    else:
        device = _resolve_device(args.device)
    print(f"PyTorch {torch.__version__} on {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPUs visible: {torch.cuda.device_count()}")
        cap = torch.cuda.get_device_capability(0)
        print(f"CUDA capability: sm_{cap[0]}{cap[1]}")
    print(f"Mode: {args.mode} (smoke={args.smoke})")
    run_battery(Path(args.out_dir), device, smoke=args.smoke, mode=args.mode)
    out_file = {
        "tuned": "results_tuned.jsonl",
        "multihop_solve": "results_multihop.jsonl",
        "multihop_deep": "results_multihop_deep.jsonl",
        "scan": "results_scan_addjump.jsonl",
        "full": "results.jsonl",
    }.get(args.mode, "results.jsonl")
    print(f"\nDone. Results in {args.out_dir}/{out_file}")


if __name__ == "__main__":
    main()
