"""Theoretical FLOPs + measured CPU wall-clock for AvA vs MHA forward.

We compute analytical FLOPs per attention layer and benchmark a forward
pass on CPU with realistic shapes. (GPU wall-clock is the more relevant
measurement; we recommend running the same script on Kaggle to extend
the table.)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ava.attention import AvaConfig, AvacchedakaAttention
from ava.model import MultiHeadSelfAttention


def ava_flops(B: int, N: int, d: int, R: int, K: int, d_a: int | None = None,
              d_r: int | None = None) -> int:
    """Theoretical multiply-adds for one AvA forward pass."""
    d_a = d_a or d // K
    d_r = d_r or d_a
    # Q/K/V projections: 3 * (B*N) * (d) * (R*K*d_r). Each of 3 einsums.
    proj = 3 * B * N * d * R * K * d_r
    # Scores: B * R * N * K * N * K matmul-style. Each entry needs d_r mults.
    scores = B * R * N * K * N * K * d_r
    # Softmax: ~5*tensor_size adds/exps; treat as B*R*N*K*N*K
    softmax = 5 * B * R * N * K * N * K
    # Weighted V: B * R * N * K * d_r aggregation, summing over (j, q):
    # for each output element, N * K mults, and N*K adds.
    weighted_v = B * R * N * K * d_r * N * K
    # Output projection W_O: B*N * (R*K * d_r * d_a)
    out_proj = B * N * R * K * d_r * d_a
    return proj + scores + softmax + weighted_v + out_proj


def mha_flops(B: int, N: int, d: int, H: int, d_h: int | None = None) -> int:
    """Theoretical multiply-adds for one MHA forward pass."""
    d_h = d_h or d // H
    # Q/K/V/O projections: 4 * B*N * d * d (each is a [d, d] linear).
    proj = 4 * B * N * d * d
    # Scores: B * H * N * d_h * N
    scores = B * H * N * N * d_h
    # Softmax: 5 * B * H * N * N
    softmax = 5 * B * H * N * N
    # Weighted V: B * H * N * d_h * N
    weighted_v = B * H * N * d_h * N
    return proj + scores + softmax + weighted_v


def measure_forward(layer, x, n_iter: int = 20):
    layer.eval()
    with torch.no_grad():
        # Warm-up
        for _ in range(3):
            _ = layer(x)
        t0 = time.time()
        for _ in range(n_iter):
            _ = layer(x)
        return (time.time() - t0) / n_iter


def benchmark(d: int, N: int, B: int, R: int, K: int, H: int):
    print(f"\n--- d={d} N={N} B={B} R={R} K={K} (vs H={H} for MHA) ---")
    ava = AvacchedakaAttention(AvaConfig(d_model=d, n_aspects=K, n_relations=R))
    mha = MultiHeadSelfAttention(d, H)
    x = torch.randn(B, N, d)
    ava_t = measure_forward(ava, x, n_iter=10)
    mha_t = measure_forward(mha, x, n_iter=10)
    ava_p = sum(p.numel() for p in ava.parameters())
    mha_p = sum(p.numel() for p in mha.parameters())
    ava_f = ava_flops(B, N, d, R, K)
    mha_f = mha_flops(B, N, d, H)
    print(f"  params         AvA={ava_p:,}     MHA={mha_p:,}     ratio={ava_p/mha_p:.2f}x")
    print(f"  theoretical FLOPs  AvA={ava_f:.2e}  MHA={mha_f:.2e}  ratio={ava_f/mha_f:.2f}x")
    print(f"  CPU wall-clock fwd  AvA={ava_t*1000:.1f}ms  MHA={mha_t*1000:.1f}ms  ratio={ava_t/mha_t:.2f}x")
    return {
        "d": d, "N": N, "B": B, "R": R, "K": K, "H": H,
        "ava_params": ava_p, "mha_params": mha_p, "param_ratio": ava_p / mha_p,
        "ava_flops": ava_f, "mha_flops": mha_f, "flops_ratio": ava_f / mha_f,
        "ava_ms": ava_t * 1000, "mha_ms": mha_t * 1000, "wall_ratio": ava_t / mha_t,
    }


def main():
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    rows = []
    # Three realistic shapes from the paper.
    rows.append(benchmark(d=128, N=15, B=128, R=4, K=4, H=4))   # M1 baseline
    rows.append(benchmark(d=256, N=22, B=128, R=4, K=4, H=4))   # 8M scale (multi-hop seq)
    rows.append(benchmark(d=512, N=22, B=128, R=4, K=4, H=4))   # 44M scale
    rows.append(benchmark(d=128, N=64, B=64, R=4, K=4, H=4))    # SCAN-like
    with open(out_dir / "flops_walltime.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {out_dir / 'flops_walltime.json'}")


if __name__ == "__main__":
    main()
