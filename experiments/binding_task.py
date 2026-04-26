"""
Conjunction binding task — a minimal diagnostic for whether an attention
mechanism can jointly bind (entity, property) -> value.

Each example is a fixed-length sequence:
    [ e1 p1 v1  e2 p2 v2  e3 p3 v3  e4 p4 v4   e_q p_q   <A> ]
The model predicts the value token at the final position. To answer, it
must find the binding whose (entity, property) matches the query *jointly*:
attending to "any matching entity" or "any matching property" is not
enough, because the context is constructed so that the query entity
appears in multiple bindings with different properties (and vice versa).

Compositional generalization split:
    - The pair space E x P is partitioned into train_pairs and test_pairs.
    - A training example draws its query pair from train_pairs; a test
      (OOD) example draws from test_pairs.
    - Context bindings for both splits draw from the full pair space, so
      every pair is observed in context during training; only the
      supervised "retrieve this pair" signal is held out.

This is the simplest test that separates compositional bind-and-retrieve
from memorized pair lookup.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class BindingSpec:
    n_entities: int = 8
    n_properties: int = 4
    n_values: int = 16
    n_bindings: int = 4          # bindings per example (context size)
    test_pair_frac: float = 0.25  # fraction of (e,p) pairs held out for OOD
    seed: int = 0


class BindingVocab:
    """Unified token vocabulary covering specials, entities, properties, values."""

    PAD = 0
    ANSWER = 1

    def __init__(self, spec: BindingSpec):
        self.spec = spec
        self.entity_offset = 2
        self.property_offset = self.entity_offset + spec.n_entities
        self.value_offset = self.property_offset + spec.n_properties
        self.vocab_size = self.value_offset + spec.n_values

    def entity_tok(self, e: int) -> int: return self.entity_offset + e
    def property_tok(self, p: int) -> int: return self.property_offset + p
    def value_tok(self, v: int) -> int: return self.value_offset + v

    @property
    def value_token_range(self) -> tuple[int, int]:
        return (self.value_offset, self.value_offset + self.spec.n_values)


def build_pair_split(spec: BindingSpec) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    rng = random.Random(spec.seed)
    all_pairs = [(e, p) for e in range(spec.n_entities) for p in range(spec.n_properties)]
    rng.shuffle(all_pairs)
    n_test = max(1, int(round(spec.test_pair_frac * len(all_pairs))))
    return all_pairs[n_test:], all_pairs[:n_test]


class BindingDataset(torch.utils.data.Dataset):
    """Generates binding examples on the fly.

    Args:
        split: "train" or "test".
            - In "weak-OOD" mode (default), context bindings are drawn from
              all (e,p) pairs; only the supervised query pair is held out
              by split.
            - In "strict-OOD" mode (`strict=True`), test pairs are entirely
              excluded from training: no training example contains a test
              pair anywhere, even as a context binding. This is a true
              compositional test --- the model has never seen the test
              pair in any role at training time.
        n_samples: nominal length.
        spec: task spec.
        strict: if True, train and test draw context bindings only from
            their respective pools.
    """

    def __init__(
        self,
        split: str,
        n_samples: int,
        spec: BindingSpec,
        train_pairs: list[tuple[int, int]],
        test_pairs: list[tuple[int, int]],
        seed: int = 0,
        strict: bool = False,
    ):
        assert split in ("train", "test")
        self.split = split
        self.n_samples = n_samples
        self.spec = spec
        self.vocab = BindingVocab(spec)
        self.train_pairs = train_pairs
        self.test_pairs = test_pairs
        self.seed = seed
        self.strict = strict

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = random.Random(f"{self.seed}-{self.split}-{idx}")
        query_pool = self.train_pairs if self.split == "train" else self.test_pairs
        e_q, p_q = rng.choice(query_pool)

        # Build context of N distinct (e,p) bindings, including the query pair.
        # Hard version: for the query's entity, include at least one OTHER property
        # in context; and for the query's property, include at least one OTHER entity.
        # This forces joint binding (a single-attribute attender fails).
        spec = self.spec
        if self.strict:
            # In strict-OOD mode, training contexts use only train_pairs;
            # test contexts use only test_pairs (plus the query pair, which
            # is in the test set).
            allowed_pairs = self.train_pairs if self.split == "train" else self.test_pairs
        else:
            allowed_pairs = self.train_pairs + self.test_pairs
        other_props_for_e_q = [(e_q, p) for p in range(spec.n_properties) if p != p_q]
        other_ents_for_p_q = [(e, p_q) for e in range(spec.n_entities) if e != e_q]
        other_props_for_e_q = [pr for pr in other_props_for_e_q if pr in allowed_pairs]
        other_ents_for_p_q = [pr for pr in other_ents_for_p_q if pr in allowed_pairs]

        chosen: list[tuple[int, int]] = [(e_q, p_q)]
        if other_props_for_e_q:
            chosen.append(rng.choice(other_props_for_e_q))
        if other_ents_for_p_q:
            chosen.append(rng.choice(other_ents_for_p_q))
        remaining = [pr for pr in allowed_pairs if pr not in chosen]
        rng.shuffle(remaining)
        while len(chosen) < spec.n_bindings and remaining:
            chosen.append(remaining.pop())
        # Truncate if we over-filled (n_bindings < 3 pathological case).
        chosen = chosen[: spec.n_bindings]
        rng.shuffle(chosen)

        # Assign values. Values are sampled fresh per example.
        values = [rng.randrange(spec.n_values) for _ in chosen]
        # The query's answer = the value assigned to the (e_q, p_q) binding.
        answer_value = values[chosen.index((e_q, p_q))]

        # Build token sequence.
        V = self.vocab
        tokens: list[int] = []
        for (e, p), v in zip(chosen, values):
            tokens += [V.entity_tok(e), V.property_tok(p), V.value_tok(v)]
        tokens += [V.entity_tok(e_q), V.property_tok(p_q), V.ANSWER]

        seq = torch.tensor(tokens, dtype=torch.long)
        target = torch.tensor(V.value_tok(answer_value), dtype=torch.long)
        # Position of the <A> token (where we read the prediction).
        answer_pos = len(tokens) - 1
        return {
            "seq": seq,
            "target": target,
            "answer_pos": torch.tensor(answer_pos, dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "seq": torch.stack([b["seq"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "answer_pos": torch.stack([b["answer_pos"] for b in batch]),
    }
