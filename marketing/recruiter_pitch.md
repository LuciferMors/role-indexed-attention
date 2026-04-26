# One-page pitch — for recruiters / hiring managers

## Rishi Shivhare — independent ML researcher

**Most recent work:** Role-Indexed Attention — a typed factorization of
transformer attention that improves compositional generalization with
reproducible measured results on a single GPU.

## Headline numbers

| Task | Multi-head attention | Role-Indexed Attention |
|------|---------------------:|-----------------------:|
| Strict-OOD binding (5 seeds) | 35.8 ± 2.6% | **99.1 ± 0.7%** |
| 44M params, strict-OOD (3 seeds) | 42.1 ± 7.3% | **98.4 ± 0.9%** |
| Multi-hop binding (depth=12) | 42.2 ± 1.5% | **69.6 ± 10.4%** (best 81.5%) |
| SCAN simple sequence-exact | 95.2 ± 1.0% | **99.7 ± 0.0%** |

All measured on a Tesla T4 GPU. All seeds documented. All code reproducible.

## What makes this strong

1. **Architecture, theorem, and experiments agree.** A separation theorem
   predicts why typed edges should beat untyped attention on conjunctive
   binding; the experimental numbers match the theorem's predictions; the
   interpretability index shows the predicted typed edges actually emerge in
   the trained model.

2. **Honest scope.** The paper documents what was tested (binding tasks at
   600K–44M params, SCAN simple split) and what was not (frontier-scale LM,
   hallucination benchmarks). I'm specifically describing the results, not
   over-claiming.

3. **Engineering quality.** The reference implementation is ~600 lines of
   PyTorch. It runs end-to-end on a laptop CPU. The full GPU battery
   reproduces on a single Tesla T4. Tests verify numerical equivalence to
   multi-head attention at K=1.

## What I'm looking for

A research engineering or research scientist position at a lab where I can
keep developing this direction:

* Scaling validation at 100M–1B parameters on natural-language
  pre-training.
* Hallucination benchmarks (TruthfulQA, HaluEval) on production-grade
  base models.
* Multi-domain transfer (vision, audio).
* Distillation experiments.

## Practical fit signals

* Independent — used to driving end-to-end ML projects from idea to
  reproducible result without supervision.
* Comfortable with PyTorch internals (custom attention modules,
  einsum-based implementations, proper init scaling).
* Comfortable with experimental rigor (multiple seeds, parameter
  matching, ablations, honest variance reporting).
* Comfortable producing publication-quality writing under time
  pressure.
* Email: rishivhare07@gmail.com
* GitHub: [link]
* arXiv: [link]
* Zenodo: [link]

## What you can ask me to do in an interview

* Walk through the AvA → MHA reduction proof.
* Explain why the joint-(j, q) softmax fails to train at K ≥ 2 and why
  per-q softmax works.
* Show the FLOPs derivation for AvA and the parameter-matching
  arithmetic.
* Implement a top-k limitor gating variant from scratch in 30 minutes.
* Explain what would break first when scaling this to 7B parameters.
