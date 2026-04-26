# Council Review — Avacchedaka Attention

Four simulated harsh NeurIPS reviewers attack the paper. Each finds the
weakest point they can. Responses describe the concrete paper change
that will address the attack. This file is the internal log; the
response actions drive the rewrite.

---

## Reviewer 1 — "The Ancient-Wisdom Sceptic"

**Attack.** The paper's appeal to Navya-Nyāya is decorative. Strip out
every mention of Sanskrit and you have: "multi-head attention with
typed edges and aspect-routed output." That's an architecture paper, not
a contribution justified by 700-year-old logic. Reviewers will see this
as Orientalist window-dressing. Borderline reject.

**Response.**
1. Cut overclaiming in the intro. Reframe Navya-Nyāya as a *source of
   inductive bias*, not as validation.
2. Move the Navya-Nyāya exposition to a single section (§2), one page,
   explicitly labelled as inspiration.
3. The method section (§3) must stand on its own as a neural
   architecture — readable and fundable with Navya-Nyāya removed.
4. Add a citation to Bhartrhari/Ganeri rigorously (refs.bib entries with
   page numbers in §2 claims).
5. Title kept, but subtitle made precise: "limitor-indexed edges for
   compositional binding."

## Reviewer 2 — "The Methodology Sceptic"

**Attack.** The experiment is one synthetic task. The claim "strictly
more expressive than MHA at matched parameters" (Thm 2) is undermined
because in the actual run, AvA has 1.05M parameters and MHA has 605k.
The claim of "matched params" in the abstract is false as presented.
Reject.

**Response.**
1. Report BOTH results honestly:
   - Same-`d_model` (AvA has more params, still compared, honest).
   - Param-matched by scaling MHA's `d_model` up to 176 (this is the
     `param_matched` battery run).
2. Re-title the table: "same d_model" vs "param-matched".
3. Rewrite Theorem 2 to be about the function class at matched params
   with a *specific* realization: AvA with aspect-factored output is
   strictly more expressive than any MHA with the same attention-layer
   parameter count.
4. Add a compute table (FLOPs / wall-clock on M1 CPU) alongside params.

## Reviewer 3 — "The Generalization Sceptic"

**Attack.** (a) The "compositional OOD" split holds out 25% of (e,p)
pairs but every pair appears in context during training. The OOD
accuracy is almost a lower bound on IID — so demonstrating a
"compositional gap" is circular. (b) One synthetic task is not enough —
run SCAN, COGS, or at least a second binding variant. Weak reject.

**Response.**
1. Add a sample-efficiency plot (the `sample_eff` battery): show AvA
   dominates MHA at all training budgets, and crucially the *gap*
   widens at low data (5k, 15k). This is the real story — compositional
   inductive bias buys sample efficiency, not just a ceiling.
2. Add a **second** task: multi-hop binding. Given `(e1, p1, eref), (eref,
   p2, v)` find `v` for query `(e1, p1, p2)`. Requires chained
   limitor-routing. Punt to appendix if time-constrained; at minimum
   describe precisely.
3. Admit limitation explicitly in §Limitations; frame current evidence
   as "controlled diagnostic + sample-efficiency + interpretability,"
   with SCAN/COGS as future work.

## Reviewer 4 — "The Theory Sceptic"

**Attack.** Theorem 1 is trivial (set `K=1`). Theorem 2 is a claim about
expressiveness but the proof sketch constructs one function; that does
not show "strict domination," only "strictly larger." Theorems are
papered over appendix placeholders. Also: the joint softmax in the
original formulation was dropped after it failed to train — the paper
must acknowledge this design choice with empirical justification, not
post-hoc rationalization. Reject.

**Response.**
1. Rewrite Theorem 2 precisely: for `K ≥ 2, R ≥ 2` and any parameter
   budget `B ≥ 4d²`, the set of attention functions realizable by AvA
   is a strict superset of those realizable by *any* MHA of the same
   budget. Give the full proof, not a sketch.
2. Explicitly discuss the two softmax modes (joint over (j,q) vs
   per-q softmax over j only). Report ablation: joint softmax does
   not train at K≥2 (this is a real finding about optimization, not
   a flaw); per-q softmax is our default. Tie to the "local vs joint
   normalization" literature.
3. Add a **Proposition** (stability / identifiability): AvA with per-q
   softmax and full-x Q/K/V recovers MHA as a strict special case
   when aspect dimension collapses (K=1, W_O[r,0] concatenated ≡
   MHA's output projection).
4. Ablate the aspect-mixer choice (we removed it; report why).

---

## Items that must land before submission

- [x] Fix init bug that silently crippled AvA for a full day of runs.
      Document in §Impl Details.
- [x] Honest parameter-matched comparison (battery: param_matched).
- [x] Sample-efficiency curves (battery: sample_eff).
- [x] Interpretability quantitative metric (specialization index in
      `experiments/interpret.py`).
- [ ] Multi-hop binding as second task.
- [ ] Rewritten §1 (intro) with no over-claiming.
- [ ] Rewritten §2 (NN background) as inspiration only, one page.
- [ ] Rewritten §3 (method) self-contained, design choices justified.
- [ ] Rewritten §4 (theory) with real proofs, no placeholders.
- [ ] §5 (experiments) with full battery results and plots.
- [ ] §6 (interpretability) with specialization index.
- [ ] §Limitations honest and specific.
- [ ] All tables filled with measured numbers.
- [ ] Compile-clean (will ship LaTeX + arXiv-compatible preamble).
