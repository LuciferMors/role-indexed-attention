"""Train AvA vs MHA at matched params on the binding task.

Usage:
    python -m experiments.train --attention ava --seed 0
    python -m experiments.train --attention mha --seed 0

Writes a JSON line per (attention, seed, step) to results/metrics.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ava import TinyTransformer, TransformerConfig
from experiments.binding_task import (
    BindingDataset,
    BindingSpec,
    BindingVocab,
    build_pair_split,
    collate,
)
from experiments.multihop_task import (
    MultiHopDataset,
    MultiHopSpec,
    MultiHopVocab,
    build_query_split,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: TinyTransformer, loader: DataLoader, device: str, value_range: tuple[int, int]) -> dict:
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    lo, hi = value_range
    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            target = batch["target"].to(device)
            pos = batch["answer_pos"].to(device)
            logits = model(seq)  # [B, N, V]
            idx = torch.arange(logits.size(0), device=device)
            ans_logits = logits[idx, pos]  # [B, V]
            # Restrict predictions to the value-token range (task-appropriate readout).
            mask = torch.full_like(ans_logits, float("-inf"))
            mask[:, lo:hi] = 0.0
            ans_logits = ans_logits + mask
            loss = F.cross_entropy(ans_logits, target, reduction="sum")
            loss_sum += loss.item()
            pred = ans_logits.argmax(dim=-1)
            correct += (pred == target).sum().item()
            total += target.numel()
    model.train()
    return {"acc": correct / total, "loss": loss_sum / total}


def run(args: argparse.Namespace) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    if args.task == "binding":
        spec = BindingSpec(seed=args.task_seed)
        train_pairs, test_pairs = build_pair_split(spec)
        vocab = BindingVocab(spec)
        strict = bool(args.strict)
        train_set = BindingDataset("train", args.n_train, spec, train_pairs, test_pairs, seed=args.seed, strict=strict)
        iid_val_set = BindingDataset("train", 2000, spec, train_pairs, test_pairs, seed=args.seed + 1000, strict=strict)
        ood_val_set = BindingDataset("test", 2000, spec, train_pairs, test_pairs, seed=args.seed + 2000, strict=strict)
    elif args.task == "multihop":
        spec = MultiHopSpec(seed=args.task_seed)
        train_q, test_q = build_query_split(spec)
        vocab = MultiHopVocab(spec)
        train_set = MultiHopDataset("train", args.n_train, spec, train_q, test_q, seed=args.seed)
        iid_val_set = MultiHopDataset("train", 2000, spec, train_q, test_q, seed=args.seed + 1000)
        ood_val_set = MultiHopDataset("test", 2000, spec, train_q, test_q, seed=args.seed + 2000)
    else:
        raise ValueError(args.task)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    iid_loader = DataLoader(iid_val_set, batch_size=256, collate_fn=collate)
    ood_loader = DataLoader(ood_val_set, batch_size=256, collate_fn=collate)

    cfg = TransformerConfig(
        vocab_size=vocab.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_aspects=args.n_aspects,
        n_relations=args.n_relations,
        d_relation=args.d_relation,
        d_aspect=args.d_aspect,
        max_seq_len=64,
        dropout=0.0,
        attention=args.attention,
    )
    model = TinyTransformer(cfg).to(device)
    n_params = model.n_parameters()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = args.n_train // args.batch_size
    warmup = max(1, int(0.05 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    results_path = Path(args.results_dir) / "metrics.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    def log(rec: dict) -> None:
        rec = {"attention": args.attention, "seed": args.seed, "n_params": n_params, **rec}
        with open(results_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    t0 = time.time()
    step = 0
    running_loss = 0.0
    running_acc = 0.0
    running_n = 0
    lo, hi = vocab.value_token_range

    for batch in train_loader:
        seq = batch["seq"].to(device)
        target = batch["target"].to(device)
        pos = batch["answer_pos"].to(device)

        logits = model(seq)
        idx = torch.arange(logits.size(0), device=device)
        ans_logits = logits[idx, pos]
        # Constrain readout to value tokens.
        mask = torch.full_like(ans_logits, float("-inf"))
        mask[:, lo:hi] = 0.0
        ans_logits = ans_logits + mask
        loss = F.cross_entropy(ans_logits, target)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        running_loss += loss.item() * target.numel()
        running_acc += (ans_logits.argmax(-1) == target).sum().item()
        running_n += target.numel()
        step += 1

        if step % args.eval_every == 0:
            iid = evaluate(model, iid_loader, device, (lo, hi))
            ood = evaluate(model, ood_loader, device, (lo, hi))
            rec = {
                "step": step,
                "train_loss": running_loss / running_n,
                "train_acc": running_acc / running_n,
                "iid_acc": iid["acc"],
                "ood_acc": ood["acc"],
                "elapsed_s": time.time() - t0,
            }
            log(rec)
            print(
                f"[{args.attention} s{args.seed}] step {step:5d}  "
                f"train_loss {rec['train_loss']:.3f} acc {rec['train_acc']:.3f}  "
                f"iid {iid['acc']:.3f}  ood {ood['acc']:.3f}  "
                f"({rec['elapsed_s']:.1f}s)"
            )
            running_loss = running_acc = running_n = 0

    # Final eval.
    iid = evaluate(model, iid_loader, device, (lo, hi))
    ood = evaluate(model, ood_loader, device, (lo, hi))
    final = {"step": step, "final_iid_acc": iid["acc"], "final_ood_acc": ood["acc"]}
    log(final)
    print(f"[{args.attention} s{args.seed}] FINAL iid {iid['acc']:.3f}  ood {ood['acc']:.3f}  "
          f"params {n_params}")
    return {"n_params": n_params, "iid": iid["acc"], "ood": ood["acc"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attention", choices=["ava", "mha"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task_seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=60000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_aspects", type=int, default=4)
    ap.add_argument("--n_relations", type=int, default=4)
    ap.add_argument("--d_relation", type=int, default=None)
    ap.add_argument("--d_aspect", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--strict", type=int, default=0,
                    help="If 1, use strict-OOD: test pairs never appear in training contexts.")
    ap.add_argument("--task", choices=["binding", "multihop"], default="binding")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
