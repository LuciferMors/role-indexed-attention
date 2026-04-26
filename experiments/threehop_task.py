"""Three-hop binding task — generalises multi-hop to k=3.

Each example contains:
    Type-A bindings: (entity, property, entity)  -- k entries chain link
    Type-B bindings: (entity, property, value)   -- terminal value
Query: (e_q, p1, p2, p3). Answer: chain through e_q -p1-> e_a -p2-> e_b
    (Type-A) and then e_b -p3-> v (Type-B). Output v.

This is harder than 2-hop: requires three composed routing steps.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class ThreeHopSpec:
    n_entities: int = 8
    n_properties: int = 4
    n_values: int = 16
    n_bindings_A: int = 4   # type-A (entity-to-entity) bindings
    n_bindings_B: int = 4   # type-B (entity-to-value)
    seed: int = 0


class ThreeHopVocab:
    PAD, SEP, EOS = 0, 1, 2
    def __init__(self, spec: ThreeHopSpec):
        self.spec = spec
        self.entity_offset = 3
        self.property_offset = self.entity_offset + spec.n_entities
        self.value_offset = self.property_offset + spec.n_properties
        self.vocab_size = self.value_offset + spec.n_values
        self.ANSWER = self.PAD  # reuse 0 as <A> marker (positions distinguish role)
        # Better: dedicated marker.
    @property
    def value_token_range(self):
        return (self.value_offset, self.value_offset + self.spec.n_values)
    def E(self, e): return self.entity_offset + e
    def P(self, p): return self.property_offset + p
    def V(self, v): return self.value_offset + v


def build_query_split(spec: ThreeHopSpec, test_frac: float = 0.25):
    rng = random.Random(spec.seed)
    triples = []
    for e in range(spec.n_entities):
        for p1 in range(spec.n_properties):
            for p2 in range(spec.n_properties):
                for p3 in range(spec.n_properties):
                    if len({p1, p2, p3}) < 3:  # require distinct hop-properties.
                        continue
                    triples.append((e, p1, p2, p3))
    rng.shuffle(triples)
    n_test = max(1, int(round(test_frac * len(triples))))
    return triples[n_test:], triples[:n_test]


class ThreeHopDataset(Dataset):
    def __init__(self, split, n_samples, spec, train_q, test_q, seed=0):
        self.split, self.n_samples, self.spec = split, n_samples, spec
        self.vocab = ThreeHopVocab(spec)
        self.train_q, self.test_q, self.seed = train_q, test_q, seed

    def __len__(self): return self.n_samples

    def __getitem__(self, idx):
        rng = random.Random(f"{self.seed}-3h-{self.split}-{idx}")
        spec = self.spec
        pool = self.train_q if self.split == "train" else self.test_q
        e_q, p1, p2, p3 = rng.choice(pool)
        # Pick the chain entities and final value.
        e_pool = [e for e in range(spec.n_entities) if e != e_q]
        e_a = rng.choice(e_pool)
        e_b_pool = [e for e in e_pool if e != e_a]
        e_b = rng.choice(e_b_pool)
        v_ans = rng.randrange(spec.n_values)

        # Required chain.
        type_A = [(e_q, p1, e_a), (e_a, p2, e_b)]
        type_B = [(e_b, p3, v_ans)]

        # Distractors -- same-property, different entity / vice versa.
        # Type-A distractors:
        for _ in range(spec.n_bindings_A - 2):
            de = rng.choice(range(spec.n_entities))
            dp = rng.choice(range(spec.n_properties))
            dt = rng.choice(range(spec.n_entities))
            type_A.append((de, dp, dt))
        # Type-B distractors:
        for _ in range(spec.n_bindings_B - 1):
            de = rng.choice(range(spec.n_entities))
            dp = rng.choice(range(spec.n_properties))
            dv = rng.randrange(spec.n_values)
            type_B.append((de, dp, dv))

        all_b = [("A", *b) for b in type_A] + [("B", *b) for b in type_B]
        rng.shuffle(all_b)

        V = self.vocab
        toks = []
        for kind, a, b, c in all_b:
            if kind == "A":
                toks += [V.E(a), V.P(b), V.E(c)]
            else:
                toks += [V.E(a), V.P(b), V.V(c)]
        # Query: e_q, p1, p2, p3, ANSWER
        toks += [V.E(e_q), V.P(p1), V.P(p2), V.P(p3), V.ANSWER]
        return {
            "seq": torch.tensor(toks, dtype=torch.long),
            "target": torch.tensor(V.V(v_ans), dtype=torch.long),
            "answer_pos": torch.tensor(len(toks) - 1, dtype=torch.long),
        }


def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
