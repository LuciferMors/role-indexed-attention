"""Causal masking tests for AvA and MHA."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ava.attention import AvaConfig, AvacchedakaAttention
from ava.model import MultiHeadSelfAttention


def test_causal_zero_future():
    """Position i must not attend to j > i."""
    torch.manual_seed(0)
    D, K, R, N = 32, 2, 2, 7
    layer = AvacchedakaAttention(AvaConfig(d_model=D, n_aspects=K, n_relations=R, causal=True))
    x = torch.randn(2, N, D)
    y, edges = layer(x, return_edges=True)
    # edges shape [B, R, N, K, N, K]. Must be zero for j > i.
    for i in range(N):
        for j in range(i + 1, N):
            mass = edges[:, :, i, :, j, :].abs().max().item()
            assert mass < 1e-6, f"Future leak at i={i} j={j}: {mass}"
    print("PASS: causal AvA never attends to future positions")


def test_causal_ava_eq_causal_mha_at_k1():
    """Causal AvA(K=1, R=H, d_r=d_h) equals causal MHA when weights are matched."""
    torch.manual_seed(1)
    D, H, N = 64, 4, 8
    mha = MultiHeadSelfAttention(D, H, causal=True)
    ava = AvacchedakaAttention(AvaConfig(d_model=D, n_aspects=1, n_relations=H, d_relation=D // H, causal=True))
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
    print(f"causal max abs diff: {diff:.2e}")
    assert diff < 1e-4, f"causal AvA(K=1) should match causal MHA, got {diff}"
    print("PASS: causal AvA(K=1) numerically equals causal MHA")


def test_topk_q_correctness():
    """topk_q=K must equal full attention (no sparsity)."""
    torch.manual_seed(2)
    D, K, R, N = 32, 4, 2, 6
    cfg_full = AvaConfig(d_model=D, n_aspects=K, n_relations=R)
    cfg_full_k = AvaConfig(d_model=D, n_aspects=K, n_relations=R, topk_q=K)
    a_full = AvacchedakaAttention(cfg_full)
    a_topk = AvacchedakaAttention(cfg_full_k)
    a_topk.load_state_dict(a_full.state_dict())
    x = torch.randn(2, N, D)
    y_full = a_full(x)
    y_topk = a_topk(x)
    diff = (y_full - y_topk).abs().max().item()
    print(f"topk_q=K diff vs full: {diff:.2e}")
    assert diff < 1e-5, f"topk_q=K should equal full, got {diff}"
    # And topk_q=1 should still produce a valid output (no NaN, finite).
    cfg_k1 = AvaConfig(d_model=D, n_aspects=K, n_relations=R, topk_q=1)
    a_k1 = AvacchedakaAttention(cfg_k1)
    a_k1.load_state_dict(a_full.state_dict())
    y_k1 = a_k1(x)
    assert torch.isfinite(y_k1).all(), "topk_q=1 produced NaN/inf"
    print("PASS: topk_q gating preserves correctness at k=K and is finite at k=1")


def test_gradient_flow_causal():
    torch.manual_seed(3)
    D, K, R, N = 32, 4, 4, 8
    layer = AvacchedakaAttention(AvaConfig(d_model=D, n_aspects=K, n_relations=R, causal=True))
    x = torch.randn(2, N, D, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for n, p in layer.named_parameters():
        assert p.grad is not None, f"{n} has no grad"
        assert torch.isfinite(p.grad).all(), f"{n} grad has NaN/inf"
    print("PASS: causal AvA gradients flow cleanly")


if __name__ == "__main__":
    test_causal_zero_future()
    test_causal_ava_eq_causal_mha_at_k1()
    test_topk_q_correctness()
    test_gradient_flow_causal()
