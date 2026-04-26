"""Train and evaluate a causal LM on SCAN.

Usage:
    python -m experiments.train_scan --attention ava --split mcd1 --seed 0
    python -m experiments.train_scan --attention mha --split simple --seed 0

Reports:
  - training loss curve
  - sequence-level exact-match accuracy on the test set (greedy decode)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ava.causal_lm import CausalLM, CausalLMConfig, masked_cross_entropy
from experiments.scan_task import (
    ScanDataset, ScanVocab, collate_pad, fetch_scan, sequence_exact_match,
)


def set_seed(s: int) -> None:
    random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def evaluate_seq_match(model: CausalLM, vocab: ScanVocab, pairs: list, max_len: int,
                        device: str, max_new: int = 80) -> float:
    """Greedy-decode each test example. Return exact-match accuracy."""
    model.eval()
    n_correct, n_total = 0, 0
    batch_pairs: list = []
    with torch.no_grad():
        for x, y in pairs:
            x_ids = [vocab.tok2id[w] for w in x.split()] + [vocab.SEP]
            y_ids = [vocab.tok2id[w] for w in y.split()] + [vocab.EOS]
            prompt = torch.tensor([x_ids], dtype=torch.long, device=device)
            if prompt.size(1) >= max_len:
                continue  # too-long input, skip (shouldn't happen with SCAN)
            out = model.greedy_decode(prompt, max_new=min(max_new, max_len - prompt.size(1)),
                                       stop_token=vocab.EOS)
            generated = out[0, prompt.size(1):].tolist()
            # Truncate at EOS inclusive.
            if vocab.EOS in generated:
                generated = generated[: generated.index(vocab.EOS) + 1]
            n_correct += int(generated == y_ids)
            n_total += 1
    model.train()
    return n_correct / max(1, n_total)


def run(args: argparse.Namespace) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    print(f"Fetching SCAN split={args.split}...")
    train_pairs, test_pairs = fetch_scan(args.split, cache_dir=args.cache_dir)
    print(f"  train: {len(train_pairs)}, test: {len(test_pairs)}")

    vocab = ScanVocab(train_pairs + test_pairs)
    # Compute max length present, with margin.
    max_len = 0
    for x, y in train_pairs + test_pairs:
        L = len(x.split()) + 1 + len(y.split()) + 1
        if L > max_len: max_len = L
    max_len = min(args.max_seq_len, max_len)
    print(f"  vocab={vocab.vocab_size}, max_len={max_len}")

    train_ds = ScanDataset(train_pairs, vocab, max_len=max_len)
    test_ds = ScanDataset(test_pairs, vocab, max_len=max_len)

    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_pad, pin_memory=pin, num_workers=0)

    cfg = CausalLMConfig(
        vocab_size=vocab.vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, n_aspects=args.n_aspects, n_relations=args.n_relations,
        max_seq_len=max_len, dropout=args.dropout, attention=args.attention,
    )
    model = CausalLM(cfg).to(device)
    n_params = model.n_parameters()
    print(f"  model: {args.attention} d={args.d_model} L={args.n_layers}  params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = args.n_epochs * len(train_loader)
    warmup = max(1, int(0.05 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup: return step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    t0 = time.time()
    step = 0
    log_path = Path(args.results_dir) / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_train_acc = 0.0
    for epoch in range(args.n_epochs):
        for batch in train_loader:
            seq = batch["seq"].to(device, non_blocking=pin)
            mask = batch["mask"].to(device, non_blocking=pin)
            # Causal LM: predict next token at each position. Loss only where
            # the target (next token) is an output token, i.e. mask shifted left.
            # logits[:, i] predicts seq[:, i+1].
            logits = model(seq[:, :-1])               # [B, N-1, V]
            targets = seq[:, 1:]                       # [B, N-1]
            loss_mask = mask[:, 1:]                    # 1 on output target positions
            loss = masked_cross_entropy(logits, targets, loss_mask)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            step += 1
            if step % args.log_every == 0:
                # Cheap proxy accuracy: token-level argmax on output positions.
                with torch.no_grad():
                    pred = logits.argmax(-1)
                    correct = ((pred == targets) * loss_mask).sum().item()
                    total = loss_mask.sum().item()
                last_train_acc = correct / max(1, total)
                print(f"  step {step}/{total_steps}  loss {loss.item():.3f}  tok-acc {last_train_acc:.3f}")
        # End-of-epoch eval (cheap subset for speed).
        n_eval = min(500, len(test_pairs))
        sample = random.Random(args.seed).sample(test_pairs, n_eval) if len(test_pairs) > n_eval else test_pairs
        seq_acc_partial = evaluate_seq_match(model, vocab, sample, max_len, device,
                                             max_new=max_len)
        print(f"  epoch {epoch+1}/{args.n_epochs}  partial seq-acc ({n_eval} ex): {seq_acc_partial:.3f}")

    # Final full-test eval.
    seq_acc = evaluate_seq_match(model, vocab, test_pairs, max_len, device, max_new=max_len)
    record = {
        "task": "scan", "split": args.split, "attention": args.attention,
        "seed": args.seed, "n_params": n_params, "d_model": args.d_model,
        "n_layers": args.n_layers, "n_aspects": args.n_aspects,
        "n_relations": args.n_relations, "n_train": len(train_pairs),
        "n_epochs": args.n_epochs, "lr": args.lr, "batch_size": args.batch_size,
        "iid_acc": last_train_acc,    # token-level on train-distribution train batches
        "ood_acc": seq_acc,           # exact-match sequence accuracy on test
        "elapsed_s": time.time() - t0,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"FINAL split={args.split} {args.attention} seed={args.seed}: seq-exact-match {seq_acc:.3f} "
          f"params={n_params:,} ({record['elapsed_s']:.0f}s)")
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attention", choices=["ava", "mha"], default="ava")
    ap.add_argument("--split", choices=["simple", "mcd1", "addjump"], default="mcd1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_aspects", type=int, default=4)
    ap.add_argument("--n_relations", type=int, default=4)
    ap.add_argument("--n_epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_seq_len", type=int, default=128)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--cache_dir", default="/tmp/scan_data")
    ap.add_argument("--results_dir", default="results/scan")
    ap.add_argument("--log_file", default="scan_results.jsonl")
    args, _ = ap.parse_known_args()
    run(args)


if __name__ == "__main__":
    main()
