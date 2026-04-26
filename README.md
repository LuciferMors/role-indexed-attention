# Role-Indexed Attention (RIA)

> Typed edges for compositional binding in transformers.

Official repository for the paper *"Role-Indexed Attention: Typed Edges for
Compositional Binding in Transformers"* (Vhare, 2026).

[Paper (PDF)](paper/main_zenodo.pdf) ·
[arXiv](https://arxiv.org/abs/PLACEHOLDER) ·
[Zenodo DOI](https://doi.org/PLACEHOLDER) ·
[Tweet thread](https://x.com/rishivhare/status/PLACEHOLDER)

---

## What is this?

Standard transformer attention computes a single scalar relevance per token
pair. That single number has to encode both *which relation* is being computed
and *which aspect of each token* participates. This conflation is part of why
language models bind facts to the wrong entity (one source of hallucination).

**Role-Indexed Attention** factors every attention edge into a typed triple
`(r, p, q)` — relation, source-aspect, target-aspect — drawn from learned
vocabularies. The structure mirrors frame semantics (Fillmore 1976), description
logic (OWL roles), and predicate-argument structure (PropBank).

## Headline results

All measured on a single Tesla T4 GPU.

| Task | MHA | RIA |
|------|------:|------:|
| Conjunction binding (strict OOD, 5 seeds) | 35.8 ± 2.6 | **99.1 ± 0.7** |
| 44M params, strict OOD (3 seeds) | 42.1 ± 7.3 | **98.4 ± 0.9** |
| Multi-hop binding L=12 (3 seeds) | 42.2 ± 1.5 | **69.6 ± 10.4** (best 81.5) |
| SCAN simple sequence-exact (3 seeds) | 95.2 ± 1.6 | **99.7 ± 0.0** |
| SCAN add_prim_jump | ≈ 0 | ≈ 0 (open problem) |

Strengths: clean win on conjunction binding tasks at multiple scales, monotonic
improvement with depth where MHA plateaus, interpretable typed edges.

Honest limitations: no frontier-scale validation, no real-LM pre-training,
fails the SCAN add_prim_jump split (consistent with all transformer baselines).
See §6 of the paper.

## Reproduce

```bash
git clone https://github.com/rishivhare/role-indexed-attention.git
cd role-indexed-attention
pip install -r requirements.txt   # torch, datasets

# Run all unit tests
python -m tests.test_reduction
python -m tests.test_causal
python -m tests.test_attention_shapes

# Reproduce the headline strict-OOD M1 result (~5 minutes on a laptop)
python -m experiments.train --attention ava --strict 1 --seed 0

# Reproduce the full GPU battery (~6 hours on a Tesla T4)
python kaggle/run_all.py --mode tuned
```

## How RIA differs from MHA, in one diagram

```
MHA attention tensor:   [B, H, N, N]               -- one scalar per (head, token-pair)
RIA attention tensor:   [B, R, N, K, N, K]         -- typed triple per token-pair

Reduction:  RIA(K=1, R=H)  ≡  MHA   (verified to 1.2e-7 in tests/test_reduction.py)
```

The full PyTorch layer is ~170 lines: see [`ava/attention.py`](ava/attention.py).

## Project layout

```
ava/                      core layer (forward, causal, top-k gating)
experiments/              tasks, training, integration scripts
kaggle/                   self-contained runner used for the GPU battery
tests/                    numerical equivalence + shape tests
paper/                    LaTeX source + compiled PDFs
results/                  measurement dumps (jsonl)
marketing/                outreach materials (tweet thread, cold-email templates)
```

## Citation

```bibtex
@article{vhare2026roleindexed,
  title   = {Role-Indexed Attention: Typed Edges for Compositional Binding in Transformers},
  author  = {Vhare, Rishi},
  journal = {arXiv preprint arXiv:PLACEHOLDER},
  year    = {2026}
}
```

## Contact

Rishi Vhare — `lucifermorsbio@gmail.com`

If you work on attention, compositional generalization, or interpretability
and want to discuss collaboration or scaling this work, please reach out.

## License

MIT — see [LICENSE](LICENSE).
