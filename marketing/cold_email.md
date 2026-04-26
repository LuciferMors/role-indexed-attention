# Cold-email templates

Send one per day, not bulk. Personalize the [BRACKETED] sections per recipient.

## Template A: senior researcher at a frontier lab

```
Subject: Typed-edge attention — would value 5 minutes of your time

Hi [first name],

I read [their specific paper or blog post — show you've done homework]
and you'll probably be the right person to ask about this.

I just published a paper introducing Role-Indexed Attention: every
attention edge carries a typed (relation, source-aspect, target-aspect)
triple instead of a single scalar. The structure is borrowed from
frame semantics and description logic.

Two results worth highlighting:

  * Strict-OOD binding (held-out (entity, property) pair absent from
    training): MHA at the matched-parameter budget reaches 36%,
    Role-Indexed Attention reaches 99% across 5 seeds.
  * SCAN simple split: 99.7% sequence-exact accuracy, a few points
    above MHA at matched compute.

Reference implementation is ~600 lines of PyTorch and the full GPU
battery completes on a single Tesla T4. Paper, code, results:
[GitHub link]

I'm writing because I'd genuinely value your read on the work. If you
have 10 minutes for a call sometime in the next few weeks, I'd be
grateful — and I'm also actively looking for a research engineering
position where I could keep building this out.

Best,
Rishi Shivhare
[email] | [github profile]
```

## Template B: hiring manager / recruiter at an AI lab

```
Subject: Independent compositional-generalization research — interested in joining your team

Hi [first name],

Brief introduction: I'm Rishi Shivhare, an independent ML researcher.
I just released a paper introducing Role-Indexed Attention, a typed
factorization of transformer attention that produces clean
compositional-generalization wins on multiple benchmarks (strict-OOD
binding, multi-hop chaining, SCAN).

Key results, all reproducible on a single Tesla T4:
  * Strict-OOD binding: 99.1% vs MHA's 35.8% at matched parameters
  * 44M params with proper recipe: 98.4% vs MHA's 42%
  * SCAN simple split: 99.7% sequence-exact

Code, paper, all results: [GitHub]
Zenodo DOI: [link]

I'd like to talk to [LAB NAME]'s research team about joining as a
research engineer or scientist. Could we set up a 30-minute call?
Happy to send code-review materials, do a coding screen, or whatever
your standard process is.

Best,
Rishi Shivhare
[email] | [github] | [linkedin]
```

## Template C: someone who recently published a related paper

```
Subject: Building on [their work]

Hi [first name],

Your paper [title] cleanly identified [specific contribution]. I
think you'll be interested in a paper I just released — Role-Indexed
Attention takes the typed-relation perspective explicitly into the
attention layer, in a direction that's complementary to your work
on [their topic].

Two findings I think speak to your interest in [topic]:
  * [specific result 1 that connects to their work]
  * [specific result 2]

Code + paper: [GitHub link]

I'd love your read on it. If we end up with overlapping interests,
happy to discuss collaboration.

Best,
Rishi
```

## Targets list (research-side)

In rough order of relevance:

| Lab | Person | Reason |
|---|---|---|
| Anthropic | Chris Olah, Tom McGrath, Catherine Olsson | Mechanistic interp angle |
| Anthropic | Sam Bowman, Jacob Steinhardt | Compositional generalization |
| Anthropic | Recruiting team | Research engineer pipeline |
| Google DeepMind | Felix Hill, Murray Shanahan | Compositional generalization, language |
| Google DeepMind | Charles Blundell, Soeren Mindermann | Architectures |
| OpenAI | Recruiting + applied research | Applied attention work |
| Meta AI / FAIR | Mike Lewis, Yann LeCun's team | Architectures |
| Mistral AI | Recruiting team | Frontier-quality engineering |
| Cohere For AI | Sara Hooker | Independent-researcher friendly |
| Microsoft Research | Sebastien Bubeck, Yi Tay | Architectures, scaling |
| Stanford CRFM | Percy Liang, Dorsa Sadigh | Compositional reasoning |
| MIT CSAIL | Jacob Andreas | Compositional language |
| EleutherAI | Stella Biderman | Independent-research community |

For each: find their most recent specific paper or post and mention it. 1 personalization paragraph + the same body. Send 1 per day. Track replies in a spreadsheet.
