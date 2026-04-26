"""Strict-OOD binding battery: AvA vs MHA at multiple seeds, with held-out
pairs that NEVER appear in training contexts.

This is the strongest compositional generalization condition: at test
time the model must bind (entity, property) pairs that were entirely
absent from training data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from experiments.run_battery import run_one  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_train", type=int, default=150000)
    ap.add_argument("--output", type=str, default="results/strict_battery.jsonl")
    args = ap.parse_args()

    cases = [
        ("mha_d128", ["--attention", "mha", "--d_model", "128", "--n_heads", "4", "--n_layers", "3"]),
        ("mha_d176_match", ["--attention", "mha", "--d_model", "176", "--n_heads", "4", "--n_layers", "3"]),
        ("ava_K4_d128", ["--attention", "ava", "--d_model", "128", "--n_aspects", "4", "--n_relations", "4", "--n_layers", "3"]),
    ]

    out_path = HERE / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists(): out_path.unlink()

    rows = []
    for tag, args_extra in cases:
        for seed in args.seeds:
            common = ["--seed", str(seed), "--n_train", str(args.n_train),
                      "--lr", "3e-3", "--eval_every", "500",
                      "--strict", "1"]
            r = run_one(args_extra + common)
            r.update({"experiment": "strict_ood", "tag": tag, "seed": seed,
                      "n_train": args.n_train})
            rows.append(r)
            print(f"[strict] {tag} seed={seed}: iid={r['iid']:.3f} ood={r['ood']:.3f} "
                  f"params={r['params']}  ({r['elapsed_s']:.1f}s)")

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
