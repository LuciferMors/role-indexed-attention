"""Shape and gradient sanity tests for AvA attention."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ava import AvaConfig, AvacchedakaAttention, TinyTransformer, TransformerConfig


def test_forward_shapes():
    cfg = AvaConfig(d_model=64, n_aspects=4, n_relations=4)
    layer = AvacchedakaAttention(cfg)
    x = torch.randn(3, 7, 64)
    y = layer(x)
    assert y.shape == x.shape, f"expected {x.shape}, got {y.shape}"
    print("PASS: forward shapes")


def test_return_edges():
    cfg = AvaConfig(d_model=32, n_aspects=2, n_relations=3)
    layer = AvacchedakaAttention(cfg)
    x = torch.randn(2, 5, 32)
    y, edges = layer(x, return_edges=True)
    assert y.shape == x.shape
    assert edges.shape == (2, 3, 5, 2, 5, 2), f"got {edges.shape}"
    # Softmax over j (dim=-2) means each (i, r, p, q) row sums to 1.
    sums = edges.sum(dim=-2)  # over j
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    print("PASS: return_edges + softmax-sum-to-1 invariant")


def test_gradient_flows():
    cfg = TransformerConfig(vocab_size=20, d_model=32, n_layers=2, n_aspects=2,
                            n_relations=2, max_seq_len=16, attention="ava")
    model = TinyTransformer(cfg)
    idx = torch.randint(0, 20, (2, 6))
    target = torch.randint(0, 20, (2,))
    logits = model(idx)[:, -1]  # last position
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_with_grad == n_total, f"only {n_with_grad}/{n_total} have grads"
    print(f"PASS: all {n_total} parameters receive gradient")


def test_n_layers_and_attention_swap():
    for kind in ["ava", "mha"]:
        cfg = TransformerConfig(vocab_size=10, d_model=32, n_layers=3, attention=kind,
                                n_aspects=2, n_relations=4, n_heads=4)
        m = TinyTransformer(cfg)
        x = torch.randint(0, 10, (1, 8))
        y = m(x)
        assert y.shape == (1, 8, 10)
    print("PASS: ava and mha both forward correctly through TinyTransformer")


if __name__ == "__main__":
    test_forward_shapes()
    test_return_edges()
    test_gradient_flows()
    test_n_layers_and_attention_swap()
