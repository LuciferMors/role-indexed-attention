"""AvA at K=1 reduces exactly to standard multi-head attention.

This is the numerical witness for Theorem 1 (MHA Subsumption). We
construct an MHA layer and an AvA(K=1, R=H, d_r=d_h) layer, copy the
parameters, and check that the outputs match within float precision.

Run with: python -m tests.test_reduction
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ava.attention import AvaConfig, AvacchedakaAttention
from ava.model import MultiHeadSelfAttention


def test_ava_k1_equals_mha(seed: int = 0, D: int = 64, H: int = 4, N: int = 8) -> None:
    torch.manual_seed(seed)
    mha = MultiHeadSelfAttention(D, H)
    ava = AvacchedakaAttention(AvaConfig(d_model=D, n_aspects=1, n_relations=H, d_relation=D // H))

    # Copy MHA's weights into AvA's typed slots.
    with torch.no_grad():
        for h in range(H):
            ava.W_Q[h, 0] = mha.W_Q.weight.T[:, h * D // H : (h + 1) * D // H]
            ava.W_K[h, 0] = mha.W_K.weight.T[:, h * D // H : (h + 1) * D // H]
            ava.W_V[h, 0] = mha.W_V.weight.T[:, h * D // H : (h + 1) * D // H]
            ava.W_O[h, 0] = mha.W_O.weight.T[h * D // H : (h + 1) * D // H, :]

    x = torch.randn(2, N, D)
    y_mha = mha(x)
    y_ava = ava(x)

    diff = (y_mha - y_ava).abs().max().item()
    print(f"max abs diff: {diff:.2e}")
    assert diff < 1e-4, f"AvA(K=1) should match MHA exactly, got diff={diff}"
    print("PASS: AvA(K=1) numerically equals MHA")


if __name__ == "__main__":
    test_ava_k1_equals_mha()
