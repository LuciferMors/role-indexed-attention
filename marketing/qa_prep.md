# Q&A preparation

Anticipate hard questions. Honest answers prepared.

## "Have you tested this at GPT-4 scale?"

No. The largest test was 44M parameters on a Tesla T4. The architectural
advantage replicates monotonically across 605K, 8M, 11M, 14M, 17M, and 44M
parameters on multiple binding-task variants and on SCAN. The power-law
fit suggests qualitatively different scaling behavior; rigorous validation
at 1B+ requires resources I do not have. That's why I'm publishing now and
looking for a team that does have those resources.

## "Why isn't this just multi-head attention with extra parameters?"

We test this. AvA at K=1 is *numerically* equivalent to MHA (we verify to
1.2 × 10⁻⁷ in `tests/test_reduction.py`). The architectural change is
specifically the K ≥ 2 case where the typed factorization adds expressive
power. We also widen MHA to match AvA's parameter count (MHA-wide variants
in every table) — MHA-wide does not catch up.

## "The variance on multi-hop is huge — σ=10pp."

Real. Best AvA seed at L=12 hits 81.5% on multi-hop; mean is 69.6% across
3 seeds. The architectural ceiling is reachable; training is not yet
fully stable in this regime. Better LR schedule and more seeds would
reduce variance. Currently a known limitation.

## "Why frame semantics and not category theory or type theory?"

Frame semantics gives the cleanest one-to-one mapping to the architectural
choice (frame = relation r, frame element = aspect type p/q). Type theory
also fits but at a higher abstraction level. I picked the formalism that
makes the architectural commitment most legible.

## "Couldn't you get the same effect with explicit type embeddings + standard MHA?"

Possibly for some tasks, but not without changing the attention mechanism
itself. Adding type embeddings as input doesn't change the structure of
the attention edge — it's still a single scalar relevance per token pair.
Our separation theorem (Theorem 3) shows that no MHA at the matched
parameter budget can express the conjunctive binding indicator without
the typed factorization at the edge. Empirically MHA-wide doesn't
catch up at any tested scale.

## "Why do you test on synthetic binding tasks instead of natural language?"

We test on both. SCAN is natural-language compositional generalization.
The synthetic tasks (binding, multi-hop) let us isolate the binding
mechanism — controlling for other factors that confound natural language
results. Both classes of evidence point the same direction.

## "Is the Indic logic angle real or marketing?"

The original draft of this work used Navya-Nyāya frame because that's
where we encountered the typed-relation factorization first. The Western
formalisms (frame semantics, description logic, semantic role labelling)
say structurally the same thing and are more familiar to ML reviewers.
The published version uses the Western references; the architectural
contribution is identical and stands on the math + experiments alone,
not on either tradition.

## "What's the one experiment that would prove you wrong?"

A 1B+ parameter language model trained with RIA on a standard
pre-training corpus, evaluated on hallucination / factual recall
benchmarks, where matched-compute MHA wins. If RIA does not transfer to
real-language scale, the architectural prior was too narrow. We can't
rule that out yet.

## "Are you going to keep working on this?"

Yes. The next experiments are: scale validation at 100M+ params, real-LM
training (TinyStories then larger), domain transfer to vision tokens,
and combining RIA with state-space models. I'm looking for a research
home that wants to fund this direction.

## "What if a frontier lab just adopts the idea without hiring you?"

Acceptable but suboptimal. The work is published openly so that's a
risk I take. The compounding value is in the next 5 papers in this
direction, which is what I bring along with the original idea. A lab
that wants the architectural direction without the person who built it
is a slower path; with me, it's a faster path.

## "Could this be a startup?"

In principle yes — typed-attention models targeting hallucination-
critical enterprise applications (legal, medical, financial). In
practice, capital + distribution + sales matter more than the
technical edge for early-stage ML startups, and at this paper's
maturity the technical evidence is too narrow to base a sale on.
First the work gets published and adopted; then the commercial path
opens.

## "Why are you doing this independently?"

Two reasons. (1) The architecture only became clear as I worked
through alternatives, and the iteration speed of independent work
suited that exploration. (2) I wanted to demonstrate I can take a
research idea from question to publication-quality artifact alone. I'd
rather collaborate going forward — solo is harder to scale.
