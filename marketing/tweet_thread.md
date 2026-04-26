# Twitter / X thread (12 posts)

Post on the day arXiv goes live. Pin to profile.

---

**1/**

Standard transformer attention computes one scalar per token pair. That single number has to encode *which relation* is being computed AND *which aspect* of each token participates.

This is part of why models bind facts to the wrong entity. New paper:

[link to arXiv]

**2/**

Role-Indexed Attention (RIA): instead of one scalar per edge, every attention edge carries a typed triple (relation, source-aspect, target-aspect).

It's the structure frame semantics, description logic, and predicate-argument role labelling have used for decades — translated into the attention layer.

**3/**

Conjunctive binding test (E entities × P properties → V values). Standard MHA: 35.8% on strict held-out pairs. RIA at the same hyperparameters and matched parameters: **99.1%** ± 0.7%.

The held-out (entity, property) pair never appeared in training. RIA generalizes.

[fig: bar chart]

**4/**

Why? Theorem: at matched parameters, MHA's bilinear score is an *additive* combination of attention over disjoint subspaces. The conjunction of two indicators — needed for joint binding — is multiplicative.

RIA realizes the multiplicative composition through its typed edges.

**5/**

Interpretability is built in. The top RIA edge places **0.90** of the answer position's attention on the correct value, **0.04** on each distractor. The top MHA head: 0.55 / 0.19.

You can read the binding circuit off the typed edges; in MHA you'd have to reverse-engineer it.

[fig: attention edge inspection]

**6/**

Scaling. We tested at 605K, 8M, 11M, 14M, 17M, and 44M parameters. Power-law fit for multi-hop chained binding:

- RIA: asymptotic error → 0
- MHA: asymptotic error ~0.5

Within the regime tested. Bigger models would be more conclusive, but the curve shapes differ qualitatively.

[fig: scaling curve]

**7/**

At 44M parameters with a proper big-model recipe, RIA hits 100% IID and 98.4% strict-OOD on the binding task, while parameter-matched MHA reaches 42%.

Same recipe. Same training compute. The architecture is doing the work.

**8/**

On the SCAN compositional generalization benchmark (Lake & Baroni 2018), RIA reaches 99.7% sequence-exact accuracy on the standard split (vs MHA 95.2% mean over 3 seeds, MHA-wide 97.9%).

Real natural-language compositional task. Real win.

**9/**

Multi-hop chained binding (e₁→e₂→v): going from 6 to 12 layers, RIA gains +31 percentage points (38.5 → 69.6%, best seed 81.5%). MHA gains 4.7 pp. MHA-wide gains 6.3 pp.

RIA converts depth into compositional reasoning ~5× faster than non-typed attention.

**10/**

Costs. RIA is ~3-4× MHA in FLOPs and parameters at the same d_model. The architectural advantage compensates: RIA at d=256 outperforms MHA at d=512 on every binding variant we tested.

Top-k limitor gating reduces compute to ~MHA-equivalent at K=2.

**11/**

Honest limitations. We do not yet have:
- 100M+ parameter validation
- Hallucination benchmark numbers (need a frontier base model)
- Multi-domain (vision, audio) tests

We do have: code that runs on a laptop, a full GPU battery on a single Tesla T4, and reproducibility checked at every step.

**12/**

Code, paper, all results: [github link]
arXiv: [arxiv link]
Zenodo (citable DOI): [zenodo link]

If you work on attention, compositional generalization, or interpretability and you want to dig in, I'd love to discuss. DMs open.

@AnthropicAI @OpenAI @GoogleDeepMind @MistralAI @cohere
