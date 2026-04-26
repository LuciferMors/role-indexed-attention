# Hacker News submission

**Title** (64 char max):
`Role-Indexed Attention: typed edges for binding in transformers`

**URL**: arXiv link (NOT GitHub — HN prefers paper / blog over repo)

**First comment** (post immediately after submission to seed discussion):

---

Author here. Quick context for HN readers:

Standard transformer attention computes a single scalar per token pair. That scalar has to encode both the relation being computed AND which aspects of the source and target participate. Multi-head attention partly factors the first. The second is absent.

This shows up empirically as "binding" failures — models lose track of which fact is attached to which entity, contributing to hallucination.

The fix is to make every attention edge carry a typed triple `(r, p, q)`: relation type, source-aspect, target-aspect. It's the same factorization frame semantics (Fillmore), description logic, and semantic role labelling have used forever — translated into the attention layer.

Headline numbers, all on a single Tesla T4 GPU:

- Strict-OOD binding (held-out (entity, property) pair never seen in training): MHA 36%, RIA 99%
- 44M-parameter strict OOD with proper big-model recipe: MHA 42%, RIA 98%
- Multi-hop chained binding: MHA gains 5 pp going L=6→12, RIA gains 31 pp
- SCAN simple split: RIA 99.7% sequence-exact

Tradeoff: RIA uses ~3-4x the parameters and FLOPs of MHA at the same d_model. A top-k limitor gating variant reduces this. The architectural advantage compensates: RIA at d=256 outperforms MHA at d=512 on every variant tested.

Honest limitations: we haven't tested at >100M params, on natural-language pre-training, or on multi-domain tasks. The paper documents what was tested and what wasn't.

Reference implementation runs on a laptop CPU. Code is on GitHub. Happy to answer questions.

---

## Variants for different mods

If front-page algorithm prefers human stories, alternative title:

`What if attention had types? A new factored attention mechanism`

If reads as too jargony:

`A typed factorization of transformer attention reduces binding errors`
