# Concrete cold emails you can copy-paste

Replace `[ARXIV]`, `[GH]`, `[ZENODO]` placeholders with your URLs after launch.

Send 3 per day. Personalize the bracketed lines based on each recipient's
recent work — that's the single most important thing you can do.

---

## To Chris Olah (Anthropic)

Subject: Typed-edge attention with built-in interpretability — would love your read

Hi Chris,

Your "Mathematical Framework for Transformer Circuits" paper is what got me
thinking about whether the typing reverse-engineered through circuit analysis
could instead be built into the architecture from the start.

I just published a paper on Role-Indexed Attention. Every attention edge
carries a typed (relation, source-aspect, target-aspect) triple instead of a
single scalar. Two findings I think you'll find interesting:

  * The typed edges spontaneously specialize: top edge places 0.90 of the
    answer position's attention on the correct value, 0.04 on each
    distractor. MHA's top head: 0.55 / 0.19. The binding circuit is
    inspectable directly from the architecture.

  * On strict-OOD compositional binding (held-out (entity, property) pairs
    never seen in training): RIA 99.1% vs MHA 35.8% across 5 seeds.

Paper: [ARXIV]
Code: [GH]
Zenodo: [ZENODO]

If you have 15 minutes for a call sometime, I'd value your read on the
mechanistic-interp implications. I'm also looking for a research-engineer
position where I can scale this work — happy to share more on that
separately.

Best,
Rishi Vhare
[email] | [github]

---

## To Felix Hill (Google DeepMind)

Subject: Compositional generalization via typed-edge attention

Hi Felix,

Your work on compositional generalization in language models has shaped how I
thought about the binding problem. I just released a paper proposing a
typed-edge factorization of attention that addresses one specific kind of
compositional failure (conjunctive binding):

  * 99.1% strict-OOD on a controlled binding diagnostic (vs MHA 35.8%, 5 seeds)
  * Multi-hop binding: +31pp gain L=6→L=12 for Role-Indexed Attention,
    +5-6pp for parameter-matched MHA
  * SCAN simple split: 99.7% vs MHA's 95.2% (3 seeds)

Honest scope: fails the SCAN add_prim_jump split (consistent with all
transformer baselines) — typed edges don't substitute for missing data, only
for missing inductive bias.

Paper: [ARXIV] · Code: [GH]

Would value your read. I'm exploring research-engineer positions and would
welcome a conversation about whether DeepMind has a relevant team.

Best,
Rishi Vhare

---

## To Sara Hooker (Cohere For AI)

Subject: Independent ML researcher — Role-Indexed Attention paper, exploring Cohere For AI Scholars

Hi Sara,

I'm an independent ML researcher who just released a paper introducing
Role-Indexed Attention — a typed factorization of transformer attention.
Headline result: 99.1% strict-OOD on a compositional binding diagnostic
versus MHA's 35.8% at matched parameters, 5 seeds.

The work is reproducible on a single Tesla T4 GPU — designed so it could be
done independently, which it was.

I'm interested in the Cohere For AI Scholars program because I want to
collaborate with researchers who'd help scale this to natural-language
pre-training (the obvious next step that I cannot do alone).

Paper: [ARXIV] · Code: [GH]

Would the Scholars program be open to applications around compositional
generalization / mechanistic interpretability? Happy to send a more detailed
research proposal if relevant.

Best,
Rishi Vhare

---

## To Stella Biderman (EleutherAI)

Subject: Role-Indexed Attention paper — independent, single-GPU reproducible

Hi Stella,

I follow EleutherAI's work and admire how the community supports
independent researchers. I just released a paper on Role-Indexed Attention
— a typed factorization of attention that improves compositional binding.

Reproduces on a single Tesla T4. Code, paper, all results public:
[ARXIV] · [GH]

Two questions:

1. Would EleutherAI's #ml-discuss or #research-discuss be a good place to
   share the work for community feedback?

2. Is there a path for an independent researcher like me to work alongside
   Eleuther on scaling this — even informally? I have the architecture and
   small-scale evidence; I lack the GPU resources for >100M-param validation.

Best,
Rishi Vhare

---

## To Sam Bowman (Anthropic, NYU)

Subject: Compositional binding in transformers — paper, would value your read

Hi Sam,

Your work on systematic generalization tests for LMs is part of why I
attempted this. New paper: Role-Indexed Attention, where every attention
edge carries a typed (relation, source-aspect, target-aspect) triple
instead of one scalar.

Strict-OOD result reviewers will probably probe: 99.1 ± 0.7% on held-out
(entity, property) pairs that never appear in training, vs MHA's 35.8 ±
2.6%. 5 seeds, parameter-matched, Tesla T4 GPU.

I also include a separation theorem: at matched attention-layer
parameters, MHA's bilinear score cannot encode the conjunctive binding
indicator, while the typed factorization can.

Paper: [ARXIV] · Code: [GH]

If you have 15 minutes I'd value your read. NeurIPS submission deadline is
soon and your perspective on the framing would be helpful.

Best,
Rishi

---

## To Jacob Andreas (MIT)

Subject: Role-Indexed Attention — frame-semantic prior into the attention layer

Hi Jacob,

Your work on grounding compositional structure in language motivated some
of the framing in my new paper, *Role-Indexed Attention: Typed Edges for
Compositional Binding in Transformers*.

Core idea: import the typed-relation factorization from frame semantics
(Fillmore), description logic, and PropBank-style SRL directly into the
attention layer. Each attention edge carries (relation, source-aspect,
target-aspect) instead of one scalar.

Specific result that may interest you: on a 2-hop chained binding task,
typed-edge attention gains +31 pp going from 6 to 12 layers; vanilla MHA
gains 5-6 pp. The architectural advantage compounds with depth.

Paper: [ARXIV] · Code: [GH]

Would welcome your read. I'm also exploring research positions and would
appreciate any thoughts on where the work might fit.

Best,
Rishi Vhare

---

## Generic template — for any researcher you want to email

Subject: [Specific paper they wrote] — would value your read on Role-Indexed Attention

Hi [Name],

[ONE SENTENCE specifically connecting their work to yours. Look at their
last 2-3 papers. If you can't find a real connection, don't send.]

I just released a paper on Role-Indexed Attention — a typed factorization
of transformer attention. Headline result: [ONE SPECIFIC NUMBER from your
results that's most relevant to their interests].

Paper: [ARXIV] · Code: [GH]

[ONE SENTENCE specific ask: "would value your read", "would love feedback
on framing", "is your group hiring research engineers", etc. Pick exactly
one.]

Best,
Rishi Vhare
[email]

---

## After sending: the follow-up rule

If no reply in 10 business days, send ONE follow-up:

> Hi [Name],
>
> Following up on my note from [date] in case it got buried. Quick recap:
> [one sentence]. Paper: [ARXIV].
>
> If now isn't the right time, no problem — happy to reconnect later.
>
> Best,
> Rishi

After that, stop. Don't send a third message.
