# Final Council Review — Avacchedaka Attention (post-results)

After running the battery, four reviewers attack the *actual* paper
(not the placeholder version). Each finds the strongest remaining
weakness.

## Reviewer 1 — "The Decisive Sceptic"

**Attack.** AvA at K=4 wins big (100% vs 74%) but uses 1.7× the
parameters of MHA. The paper's claim is at "same d_model" which
*sounds* matched but isn't. The param-matched MHA d=176 then collapses
to 48% — but you didn't tune the LR for it. A 1-line LR sweep for the
matched MHA could close the gap. Without that, the matched comparison
is a strawman. Borderline reject.

**Action.**
- Document the LR=3e-3 choice and that we did not tune per-config.
- Add a sentence in §Limitations noting that LR/optimizer tuning was
  not done per-architecture; reporting at the same hyperparameters
  is intentional for fairness but means the matched-MHA number is a
  *minimum* over configurations we did not search.
- Note that *both* matched and same-d_model comparisons should be
  read together: the same-d_model comparison gives MHA the MORE-
  capable hyperparameters; even there it loses.

## Reviewer 2 — "The Variance Sceptic"

**Attack.** MHA's accuracy at d=128 is `74.7 ± 20.1%`. That's enormous
seed variance — one seed got 95%, another got 55%. With three seeds
this is not a confident measurement. AvA's ±0.0 looks deterministic
but with three seeds you can't distinguish 100.0% from 99.7%. Run more
seeds or report bootstrapped CIs. Weak reject.

**Action.**
- Acknowledge in §Experiments that the high variance of MHA on this
  task is itself a finding (bimodal: either it learns the binding
  basin or it doesn't).
- Five seeds would be better; we will run two more if time permits;
  if not, flag this in §Limitations.
- AvA's deterministic 100.0% across 3 seeds for K∈{2,4} *does* give
  high confidence on the difference; we are not arguing 100.0 vs
  99.7 here.

## Reviewer 3 — "The Sample-Efficiency Sceptic"

**Attack.** Your sample-efficiency table shows MHA *beating* AvA at
5k samples (28.0 vs 21.6). The abstract previously claimed AvA
dominates at low budgets (it doesn't — AvA underfits with its larger
parameter count). The table also shows **both** methods stuck at the
same plateau through 50k, with the win only emerging at 150k. The
"sample efficiency" framing is misleading — the actual phenomenon is
a phase transition, not faster learning. Reject the framing.

**Action — done.**
- Removed the "AvA at 5k beats MHA at 30× the data" abstract claim.
- Reframed as a phase transition: both are stuck at the weak-attention
  plateau until the binding circuit can fit; AvA crosses the threshold
  by 150k, MHA does not.
- Added explicit caveat that AvA is *worse* than MHA at the smallest
  budget due to underfitting.

## Reviewer 4 — "The Generalization Sceptic"

**Attack.** Single synthetic task. No SCAN, COGS, ListOps, no real
language data. The compositional split holds out 8 of 32 (e,p) pairs,
but every pair appears in the context — the held-out signal is only
on the *query*, which is a relatively weak compositional test. You
cannot extrapolate from this to "transformers at scale would benefit."
Reject as out of scope for NeurIPS without natural-language results.

**Action — partial.**
- Acknowledge as the headline limitation: §Limitations now states
  this explicitly.
- Add scope statement in §Experiments: "We are establishing the
  *mechanism*, not the scaling. SCAN, COGS, and language modeling are
  the obvious next experiments."
- Argue (in §Conclusion) the contribution is (a) the architecture
  itself, (b) the controlled mechanism evidence, and (c) interpretability.

---

## Likelihood of acceptance

The paper is now coherent and defensible at the workshop level.
For a NeurIPS main-track outcome with high probability, the bare
minimum next step is a **second task** — even something as simple as
ListOps or a multi-hop binding variant. If that also shows AvA winning
with the same circuit-level interpretation, the paper becomes hard to
reject.
