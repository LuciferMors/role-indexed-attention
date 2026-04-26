"""Interpretability analysis: do AvA's (r, p, q) edges correspond to the task's
ground-truth roles (entity-match, property-match, value-retrieval), while
MHA's heads are anonymous?

Train one AvA and one MHA model, then:
  1. Measure each edge type's (or head's) average attention mass on
     structurally distinguished position pairs:
       - entity -> entity matches
       - property -> property matches
       - answer <- value positions
  2. Compute a "role specialization index": max/mean ratio of the mass
     an edge/head places on the intended structural pair over non-intended.
     Higher = more specialized.

Also dumps attention heatmaps for a few examples.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ava import TinyTransformer, TransformerConfig
from ava.attention import AvacchedakaAttention
from ava.model import MultiHeadSelfAttention
from experiments.binding_task import (
    BindingDataset, BindingSpec, BindingVocab, build_pair_split, collate,
)
from experiments.train import set_seed


def train_model(attention: str, n_train: int, seed: int, **cfg_over) -> TinyTransformer:
    device = "cpu"
    set_seed(seed)
    spec = BindingSpec(seed=0)
    train_pairs, test_pairs = build_pair_split(spec)
    vocab = BindingVocab(spec)
    train_set = BindingDataset("train", n_train, spec, train_pairs, test_pairs, seed=seed)
    loader = DataLoader(train_set, batch_size=128, collate_fn=collate)
    cfg = TransformerConfig(
        vocab_size=vocab.vocab_size, d_model=128, n_layers=3, n_heads=4,
        n_aspects=4, n_relations=4, max_seq_len=64, attention=attention, **cfg_over,
    )
    model = TinyTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01)
    total = n_train // 128
    warmup = max(1, int(0.05 * total))

    def lr_lambda(step):
        if step < warmup: return step / warmup
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    lo, hi = vocab.value_token_range
    step = 0
    for batch in loader:
        logits = model(batch["seq"])
        idx = torch.arange(logits.size(0))
        al = logits[idx, batch["answer_pos"]]
        mask = torch.full_like(al, float("-inf")); mask[:, lo:hi] = 0.0
        al = al + mask
        loss = F.cross_entropy(al, batch["target"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
    return model, vocab, spec


def extract_attention(model: TinyTransformer, seq: torch.Tensor) -> list[torch.Tensor]:
    """Run one forward and grab attention tensors from every block."""
    captured = []
    hooks = []

    def ava_hook(module, inp, out):
        # AvA caches the most recent attention tensor on the module.
        captured.append(module.last_alpha)

    def mha_hook(module, inp, out):
        # Recompute to get alpha. MHA's forward didn't return it; patch here.
        x, = inp
        B, N, D = x.shape
        H, dh = module.n_heads, module.head_dim
        q = module.W_Q(x).view(B, N, H, dh).transpose(1, 2)
        k = module.W_K(x).view(B, N, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        alpha = F.softmax(scores, dim=-1)  # [B, H, N, N]
        captured.append(alpha.detach())

    for blk in model.blocks:
        if isinstance(blk.attn, AvacchedakaAttention):
            hooks.append(blk.attn.register_forward_hook(ava_hook))
        elif isinstance(blk.attn, MultiHeadSelfAttention):
            hooks.append(blk.attn.register_forward_hook(mha_hook))

    with torch.no_grad():
        model(seq)
    for h in hooks: h.remove()
    return captured


def role_specialization(model, vocab: BindingVocab, spec: BindingSpec, n_examples: int = 512):
    """Quantify how strongly each AvA edge / MHA head routes the answer
    position to the correct value position.

    For each example we know which of the 4 context value positions
    is the correct answer (the one whose binding's (e, p) matches the
    query). For each edge/head we compute the mean attention from the
    answer position (i=14) to:
      - correct  : the matching value position
      - distractor: each of the 3 other value positions
      - other    : everything else (positions 0-11 entities/properties + 12-13 query)
    A "retrieval-specialized" edge has correct >> distractor + other.

    Specialization index for an edge: correct / (correct+distractor+other) -- the
    fraction of answer-position attention mass that lands on the correct
    value. Random-attention baseline is 1/15 = 0.067; uniform-over-values
    is 1/4 = 0.25; perfect binding is 1.0.
    """
    train_pairs, test_pairs = build_pair_split(spec)
    ds = BindingDataset("train", n_examples, spec, train_pairs, test_pairs, seed=999)
    loader = DataLoader(ds, batch_size=n_examples, collate_fn=collate)
    batch = next(iter(loader))
    seq = batch["seq"]  # [B, N=15]
    target_value = batch["target"]  # [B] -- the correct value token

    # Identify the "correct" value position per example. The context has 4
    # bindings at positions (e1, p1, v1) = (0, 1, 2), etc. The matching
    # binding's value position is the one whose value token equals the target.
    answer_pos = 14
    value_positions = [2, 5, 8, 11]
    B = seq.shape[0]
    correct_pos = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        for v in value_positions:
            if seq[b, v].item() == target_value[b].item():
                correct_pos[b] = v
                break

    attn_tensors = extract_attention(model, seq)
    out = []
    for L, tens in enumerate(attn_tensors):
        if tens.dim() == 6:  # AvA: [B, R, N, K, N, K]
            B_, R, N_, K, _, _ = tens.shape
            # Sum over target-aspect q to get attention from (i, p) to j.
            # Mean over q (target aspect): per_q softmax gives K distributions
            # summing to 1 each, so the marginal-over-q attention for fixed
            # (i, r, p) is comparable to a single MHA-style distribution.
            tens_marg = tens.mean(dim=-1)  # [B, R, N, K, N]
            answer_attn = tens_marg[:, :, answer_pos, :, :]  # [B, R, K, N]
            for r in range(R):
                for p in range(K):
                    a = answer_attn[:, r, p]  # [B, N]
                    correct = a[torch.arange(B_), correct_pos].mean().item()
                    distractors = []
                    for v in value_positions:
                        is_distractor = (correct_pos != v)
                        d = a[is_distractor, v].mean().item() if is_distractor.any() else 0.0
                        distractors.append(d)
                    distractor = sum(distractors) / 3
                    other = (a.sum(dim=-1).mean().item() - correct - distractor * 3)
                    out.append({"layer": L, "kind": "ava", "r": r, "p": p,
                                "correct": correct, "distractor": distractor, "other": other})
        else:  # MHA: [B, H, N, N]
            B_, H, _, _ = tens.shape
            for h in range(H):
                a = tens[:, h, answer_pos]  # [B, N]
                correct = a[torch.arange(B_), correct_pos].mean().item()
                distractors = []
                for v in value_positions:
                    is_distractor = (correct_pos != v)
                    d = a[is_distractor, v].mean().item() if is_distractor.any() else 0.0
                    distractors.append(d)
                distractor = sum(distractors) / 3
                other = (a.sum(dim=-1).mean().item() - correct - distractor * 3)
                out.append({"layer": L, "kind": "mha", "h": h,
                            "correct": correct, "distractor": distractor, "other": other})
    return out


def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    n_train = 150000
    print(f"Training AvA ({n_train} samples)...")
    ava_model, vocab, spec = train_model("ava", n_train=n_train, seed=0)
    print(f"Training MHA ({n_train} samples)...")
    mha_model, _, _ = train_model("mha", n_train=n_train, seed=0)

    ava_rows = role_specialization(ava_model, vocab, spec)
    mha_rows = role_specialization(mha_model, vocab, spec)

    # Retrieval-specialization index: fraction of answer-position attention
    # that lands on the correct value position. Random baseline is 1/15 ~ 0.067;
    # uniform-over-values is 1/4 = 0.25; perfect binding is 1.0.
    def spec_index(r):
        return r["correct"]

    for r in ava_rows + mha_rows:
        r["specialization"] = spec_index(r)

    # Summary statistics.
    ava_max = max(r["specialization"] for r in ava_rows)
    mha_max = max(r["specialization"] for r in mha_rows)
    ava_mean = sum(r["specialization"] for r in ava_rows) / len(ava_rows)
    mha_mean = sum(r["specialization"] for r in mha_rows) / len(mha_rows)
    ava_top3 = sorted(ava_rows, key=lambda r: -r["specialization"])[:3]
    mha_top3 = sorted(mha_rows, key=lambda r: -r["specialization"])[:3]

    summary = {
        "n_edges_ava": len(ava_rows),
        "n_heads_mha": len(mha_rows),
        "mean_correct_attn_ava": ava_mean,
        "mean_correct_attn_mha": mha_mean,
        "max_correct_attn_ava": ava_max,
        "max_correct_attn_mha": mha_max,
        "top3_ava": ava_top3,
        "top3_mha": mha_top3,
    }

    with open(results_dir / "interpretability.json", "w") as f:
        json.dump({"summary": summary, "ava_rows": ava_rows, "mha_rows": mha_rows}, f, indent=2)

    print("\n=== Retrieval-specialisation index (P(answer -> correct value)) ===")
    print(f"AvA edges (n={len(ava_rows)}):  max={ava_max:.3f}  mean={ava_mean:.3f}")
    print(f"MHA heads (n={len(mha_rows)}): max={mha_max:.3f}  mean={mha_mean:.3f}")
    print("\nTop-3 AvA edges by retrieval mass:")
    for r in ava_top3:
        print(f"  L{r['layer']} r={r['r']} p={r['p']}: correct={r['correct']:.3f} distractor={r['distractor']:.3f} other={r['other']:.3f}")
    print("\nTop-3 MHA heads by retrieval mass:")
    for r in mha_top3:
        print(f"  L{r['layer']} h={r['h']}: correct={r['correct']:.3f} distractor={r['distractor']:.3f} other={r['other']:.3f}")


if __name__ == "__main__":
    main()
