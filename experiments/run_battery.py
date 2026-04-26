"""Master experiment runner.

Runs the full evaluation battery:
  1. Main comparison: AvA vs MHA at same d_model and at matched params.
  2. Sample efficiency: accuracy vs training-example budget.
  3. Ablations: K in {1,2,4}, softmax mode, aspect mixer.

All runs are CPU-friendly (tested on M1 8GB). Writes to results/battery.jsonl.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def run_one(extra_args: list[str]) -> dict:
    cmd = [sys.executable, "-m", "experiments.train"] + extra_args
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    dt = time.time() - t0
    out = proc.stdout + proc.stderr
    final_line = [l for l in out.splitlines() if "FINAL" in l]
    if not final_line:
        print(out[-500:])
        raise RuntimeError(f"No FINAL line in run: {cmd}")
    # parse "FINAL iid 0.xxx  ood 0.xxx  params nnnnnn"
    parts = final_line[-1].split()
    iid = float(parts[parts.index("iid") + 1])
    ood = float(parts[parts.index("ood") + 1])
    params = int(parts[parts.index("params") + 1])
    return {"iid": iid, "ood": ood, "params": params, "elapsed_s": dt}


def experiment_main_comparison(seeds: list[int], n_train: int) -> list[dict]:
    """AvA K=4 vs MHA, same d_model=128."""
    cases = [
        ("mha_d128", ["--attention", "mha", "--d_model", "128", "--n_heads", "4", "--n_layers", "3"]),
        ("ava_K4_d128", ["--attention", "ava", "--d_model", "128", "--n_aspects", "4", "--n_relations", "4", "--n_layers", "3"]),
        ("ava_K2_d128", ["--attention", "ava", "--d_model", "128", "--n_aspects", "2", "--n_relations", "4", "--n_layers", "3"]),
        ("ava_K1_d128", ["--attention", "ava", "--d_model", "128", "--n_aspects", "1", "--n_relations", "4", "--d_relation", "32", "--n_layers", "3"]),
    ]
    rows = []
    for tag, args in cases:
        for seed in seeds:
            common = ["--seed", str(seed), "--n_train", str(n_train), "--lr", "3e-3", "--eval_every", "500"]
            r = run_one(args + common)
            r.update({"experiment": "main", "tag": tag, "seed": seed, "n_train": n_train})
            rows.append(r)
            print(f"[main] {tag} seed={seed}: iid={r['iid']:.3f} ood={r['ood']:.3f} params={r['params']}  ({r['elapsed_s']:.1f}s)")
    return rows


def experiment_param_matched(seeds: list[int], n_train: int) -> list[dict]:
    """Match MHA param count to AvA K=4 by widening d_model."""
    # AvA K=4 full-x: ~1.05M params. MHA with d_model=192 has:
    #   attn 4*192² = 147456 per layer * 3 = 442k
    #   mlp 2*192*768 per layer * 3 = 885k
    #   embed 30*192 + 64*192 ~ 18k
    # Total ~1.34M. Close enough; or use d=176: attn 4*30976*3 = 372k, mlp 595k, tot ~985k.
    cases = [
        ("mha_d176_match", ["--attention", "mha", "--d_model", "176", "--n_heads", "4", "--n_layers", "3"]),
        ("ava_K4_d128", ["--attention", "ava", "--d_model", "128", "--n_aspects", "4", "--n_relations", "4", "--n_layers", "3"]),
    ]
    rows = []
    for tag, args in cases:
        for seed in seeds:
            common = ["--seed", str(seed), "--n_train", str(n_train), "--lr", "3e-3", "--eval_every", "500"]
            r = run_one(args + common)
            r.update({"experiment": "param_matched", "tag": tag, "seed": seed, "n_train": n_train})
            rows.append(r)
            print(f"[matched] {tag} seed={seed}: iid={r['iid']:.3f} ood={r['ood']:.3f} params={r['params']}  ({r['elapsed_s']:.1f}s)")
    return rows


def experiment_sample_efficiency(seeds: list[int]) -> list[dict]:
    """Vary training-example budget."""
    budgets = [5000, 15000, 50000, 150000]
    rows = []
    for n_train in budgets:
        for tag, args in [
            ("mha_d128", ["--attention", "mha", "--d_model", "128", "--n_heads", "4", "--n_layers", "3"]),
            ("ava_K4_d128", ["--attention", "ava", "--d_model", "128", "--n_aspects", "4", "--n_relations", "4", "--n_layers", "3"]),
        ]:
            for seed in seeds:
                common = ["--seed", str(seed), "--n_train", str(n_train), "--lr", "3e-3",
                          "--eval_every", str(max(50, n_train // 128 // 3))]
                r = run_one(args + common)
                r.update({"experiment": "sample_eff", "tag": tag, "seed": seed, "n_train": n_train})
                rows.append(r)
                print(f"[sample_eff n={n_train}] {tag} seed={seed}: iid={r['iid']:.3f} ood={r['ood']:.3f}  ({r['elapsed_s']:.1f}s)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_train", type=int, default=150000)
    ap.add_argument("--experiments", type=str, nargs="+",
                    default=["main", "param_matched", "sample_eff"])
    ap.add_argument("--output", type=str, default="results/battery.jsonl")
    args = ap.parse_args()

    out_path = HERE / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    all_rows: list[dict] = []
    if "main" in args.experiments:
        all_rows += experiment_main_comparison(args.seeds, args.n_train)
    if "param_matched" in args.experiments:
        all_rows += experiment_param_matched(args.seeds, args.n_train)
    if "sample_eff" in args.experiments:
        all_rows += experiment_sample_efficiency(args.seeds)

    with open(out_path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
