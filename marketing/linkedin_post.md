# LinkedIn post (longer-form, professional audience)

---

**Role-Indexed Attention: typed edges for compositional binding in transformers**

Sharing the result of a deep dive into one of the more stubborn problems in transformer architectures.

**The problem.** Transformer attention computes a single scalar relevance per token pair. That single number has to capture both *which relation* is being computed AND *which aspect of each token* participates. Multi-head attention partly addresses the first; the second is structurally absent. This is part of why language models hallucinate by binding facts to the wrong entity.

**The fix.** Replace the scalar attention edge with a typed triple (r, p, q): a relation type r, a source-aspect type p, a target-aspect type q. This is the same factorization that frame semantics (Fillmore 1976), description logic (OWL roles), and PropBank-style semantic role labelling have used for decades to represent relational structure. We translate it into the attention layer.

**The numbers.**

| Task | MHA | Role-Indexed Attention |
|---|---:|---:|
| Conjunction binding (strict OOD) | 35.8% | **99.1%** |
| 44M params, strict OOD | 42% | **98.4%** |
| Multi-hop binding (depth=12) | 42% | **70%** (best seed 82%) |
| SCAN simple split | 95.2% | **99.7%** |

All numbers measured on a single Tesla T4 GPU, full reproducible battery completes in a few hours. Reference implementation runs end-to-end on a laptop CPU.

**Why I'm sharing this.**

I worked on this independently. The paper covers (i) the architecture, (ii) a separation theorem, (iii) a reproducible experimental battery, (iv) an interpretability index showing the typed edges spontaneously specialize to ground-truth roles, and (v) honest limitations.

Code, results, and both paper versions (anonymous NeurIPS submission + named Zenodo deposit) are public:

🔗 GitHub: [link]
🔗 Zenodo (DOI): [link]
🔗 arXiv preprint: [link]

If you work on attention mechanisms, compositional generalization, or mechanistic interpretability — at a frontier lab or otherwise — and you want to discuss this work, my contact is in the GitHub README. I'm actively looking for the right team to keep building this with.

#machinelearning #attention #compositionalgeneralization #transformers #ai
