"""Two-hop binding task.

Each example contains two kinds of bindings interleaved in context:
    Type-A:  (entity, property, entity)   -- entity has property pointing
                                             to *another entity*.
    Type-B:  (entity, property, value)    -- entity has property with a value.

Query: (e_q, p1, p2). The answer is computed by chaining:
    1. find a Type-A binding (e_q, p1, e_mid) in context;
    2. find a Type-B binding (e_mid, p2, v) in context;
    3. output v.

The model must compose two attention hops. Both hops share the same
relation ``has-property'' but differ in the type of the third token.
A flat attention (single softmax over context) cannot solve this; two
typed routings can.

Sequence layout (same vocabulary as binding_task, with the value tokens
range repurposed as needed):
    [ a1_e a1_p a1_t  a2_e a2_p a2_t  ...  e_q p1 p2 <A> ]

Distractors:
    - Type-A distractor: (e_q, p_other, e_other2)  -- same entity, wrong p1
    - Type-B distractor: (e_other, p2, v_other)    -- wrong entity, right p2
    - Plus filler bindings to dilute.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class MultiHopSpec:
    n_entities: int = 8
    n_properties: int = 4
    n_values: int = 16
    n_bindings_A: int = 3   # type-A bindings in context
    n_bindings_B: int = 3   # type-B bindings in context
    seed: int = 0


class MultiHopVocab:
    """Single shared vocabulary for entity, property, and value tokens."""

    PAD = 0
    ANSWER = 1

    def __init__(self, spec: MultiHopSpec):
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


class MultiHopDataset(torch.utils.data.Dataset):
    """Generates two-hop binding examples on the fly.

    Args:
        split: "train" | "test". Train and test draw from disjoint sets of
            ``(e_q, p1, p2)`` query triples; the second-hop binding's value
            is freshly randomized per example.
        n_samples: nominal length.
    """

    def __init__(
        self,
        split: str,
        n_samples: int,
        spec: MultiHopSpec,
        train_queries: list[tuple[int, int, int]],
        test_queries: list[tuple[int, int, int]],
        seed: int = 0,
    ):
        assert split in ("train", "test")
        self.split = split
        self.n_samples = n_samples
        self.spec = spec
        self.vocab = MultiHopVocab(spec)
        self.train_queries = train_queries
        self.test_queries = test_queries
        self.seed = seed

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = random.Random(f"{self.seed}-{self.split}-{idx}")
        spec = self.spec

        pool = self.train_queries if self.split == "train" else self.test_queries
        e_q, p1, p2 = rng.choice(pool)

        # Pick the intermediate entity (not e_q).
        e_mid_pool = [e for e in range(spec.n_entities) if e != e_q]
        e_mid = rng.choice(e_mid_pool)
        # Pick the answer value.
        v_answer = rng.randrange(spec.n_values)

        # Build the canonical answer chain.
        type_A_bindings: list[tuple[int, int, int]] = [(e_q, p1, e_mid)]
        type_B_bindings: list[tuple[int, int, int]] = [(e_mid, p2, v_answer)]

        # Add type-A distractors: (e_q, p_other, e_other) and (e_other, p1, e_other2).
        other_props = [p for p in range(spec.n_properties) if p != p1]
        if other_props:
            p_other = rng.choice(other_props)
            e_other_target = rng.choice(e_mid_pool)
            type_A_bindings.append((e_q, p_other, e_other_target))
        e_other = rng.choice([e for e in range(spec.n_entities) if e not in (e_q, e_mid)])
        e_third = rng.choice([e for e in range(spec.n_entities) if e != e_other])
        type_A_bindings.append((e_other, p1, e_third))

        # Type-B distractors: (e_other, p2, v_other), (e_mid, p_other, v_other2).
        v_other = rng.randrange(spec.n_values)
        if v_other == v_answer: v_other = (v_other + 1) % spec.n_values
        type_B_bindings.append((e_other, p2, v_other))
        if other_props:
            v_other2 = rng.randrange(spec.n_values)
            type_B_bindings.append((e_mid, rng.choice(other_props), v_other2))

        # Truncate to spec sizes.
        type_A_bindings = type_A_bindings[: spec.n_bindings_A]
        type_B_bindings = type_B_bindings[: spec.n_bindings_B]

        # Interleave and shuffle.
        all_bindings = [("A", *b) for b in type_A_bindings] + [("B", *b) for b in type_B_bindings]
        rng.shuffle(all_bindings)

        V = self.vocab
        tokens: list[int] = []
        for kind, a, b, c in all_bindings:
            if kind == "A":
                # entity, property, entity
                tokens += [V.entity_tok(a), V.property_tok(b), V.entity_tok(c)]
            else:
                # entity, property, value
                tokens += [V.entity_tok(a), V.property_tok(b), V.value_tok(c)]
        # Query: e_q, p1, p2, ANSWER
        tokens += [V.entity_tok(e_q), V.property_tok(p1), V.property_tok(p2), V.ANSWER]

        seq = torch.tensor(tokens, dtype=torch.long)
        target = torch.tensor(V.value_tok(v_answer), dtype=torch.long)
        answer_pos = len(tokens) - 1
        return {
            "seq": seq,
            "target": target,
            "answer_pos": torch.tensor(answer_pos, dtype=torch.long),
        }


def build_query_split(spec: MultiHopSpec, test_frac: float = 0.25) -> tuple[list, list]:
    """Partition (e_q, p1, p2) triples into train and test pools.

    p1 != p2 enforced so the two hops carry different property slots.
    """
    rng = random.Random(spec.seed)
    triples = []
    for e in range(spec.n_entities):
        for p1 in range(spec.n_properties):
            for p2 in range(spec.n_properties):
                if p1 == p2: continue
                triples.append((e, p1, p2))
    rng.shuffle(triples)
    n_test = max(1, int(round(test_frac * len(triples))))
    return triples[n_test:], triples[:n_test]


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "seq": torch.stack([b["seq"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "answer_pos": torch.stack([b["answer_pos"] for b in batch]),
    }
