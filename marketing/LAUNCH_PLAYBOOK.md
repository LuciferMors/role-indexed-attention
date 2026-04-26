# Launch playbook — Day-by-day

Total span: 14 days. Execute in order. Don't skip steps.

---

## Day -3 to -1 (preparation)

### Pre-flight checklist

- [ ] Repo cleaned: `.gitignore`, `LICENSE`, polished `README.md` ✅
- [ ] `paper/main_zenodo.pdf` final (named author)
- [ ] `paper/main_neurips.pdf` final (anonymous, for NeurIPS submission)
- [ ] All tests pass: `make test` from project root
- [ ] Battery results all in `results/` (don't re-run before launch — freeze them)
- [ ] Twitter/X profile filled out: bio mentions ML research; one pinned tweet about the paper draft you can reference; profile pic
- [ ] Personal website deployed at a stable URL (rishivhare.com or github.io page)
- [ ] Google Scholar profile created
- [ ] Email signature includes paper link (after launch)

### Accounts to create now

1. **Zenodo** — https://zenodo.org/signup (free, instant DOI)
2. **arXiv** — https://arxiv.org/user/register (need cs.LG endorsement; see workaround below)
3. **OpenReview** (for NeurIPS) — https://openreview.net/signup
4. **GitHub** — make repo public 1 day before launch

### arXiv endorsement workaround

If you don't have a cs.LG endorsement:
- Email a researcher who has published in cs.LG (someone you know loosely, or polite cold-email)
- Mention: "I'd appreciate cs.LG endorsement for my submission. Paper attached. Happy to reciprocate."
- Most respond if the paper is genuinely well-written.

Alternatively: submit to **cs.AI** (sometimes easier endorsement) or **cs.NE** (neural and evolutionary computing). Cross-list to cs.LG once endorsed.

If endorsement takes >1 week: **publish on Zenodo first** (no endorsement needed), get the DOI, use the DOI for outreach. arXiv can come a few days later.

---

## Day 1 — Zenodo deposit (the public-record anchor)

This is **the** first thing you do. Zenodo gives you a permanent DOI, citable forever, no gatekeeping.

### Step-by-step

1. Log in at https://zenodo.org
2. Click **"New upload"**
3. Upload files:
   - `paper/main_zenodo.pdf` (the named version)
   - `paper/main_zenodo.tex` (source)
   - `paper/refs.bib`
   - Code archive: `git archive --format=zip HEAD > role-indexed-attention-v1.0.zip`
4. Fill metadata:
   - **Type:** Publication → Preprint
   - **Title:** Role-Indexed Attention: Typed Edges for Compositional Binding in Transformers
   - **Authors:** Rishi Vhare (Independent researcher), ORCID if you have one
   - **Description:** paste your abstract verbatim
   - **License:** MIT for code, CC-BY-4.0 for paper
   - **Keywords:** `attention`, `transformers`, `compositional generalization`, `frame semantics`, `binding`, `interpretability`
   - **Communities:** "Open Science"
5. Click **Publish** — DOI is issued immediately, looks like `10.5281/zenodo.XXXXXXX`

### What you get

- Permanent URL `https://doi.org/10.5281/zenodo.XXXXXXX`
- DOI you can cite, link from website, put on LinkedIn
- Cannot be retracted (good — establishes priority)

### Before submitting: triple-check
- Author name spelled correctly
- Email correct
- All figures embedded in PDF
- Code archive is complete (test by extracting it and running smoke tests)

---

## Day 2 — arXiv submission

Once Zenodo is up, file the arXiv paper.

### Step-by-step

1. Submit at https://arxiv.org/submit
2. **Primary category:** cs.LG (machine learning)
3. **Secondary categories:** cs.CL (computation and language), cs.AI
4. Upload: prepare a single `.tar.gz` with `main.tex`, `refs.bib`, and figure files
5. **Title:** same as Zenodo
6. **Abstract:** paste from paper (under 1920 chars; trim if needed)
7. **Comments field:** "Code and additional results: github.com/rishivhare/role-indexed-attention. Zenodo: doi.org/10.5281/zenodo.XXXXXXX"
8. **License:** arXiv non-exclusive license (default)

Submission queues for moderation, takes 1-2 business days. Expect Mon/Tue announce.

---

## Day 3 — GitHub public, Twitter thread, LinkedIn

### Morning (your timezone): Make repo public

```bash
cd /Users/rishi/Desktop/y/avacchedaka
# Final touchup: ensure paper PDF is in repo
git add paper/main_zenodo.pdf paper/main_neurips.pdf README.md LICENSE
git commit -m "Public release"
git remote add origin https://github.com/rishivhare/role-indexed-attention.git
git push -u origin main
```

Verify:
- README renders correctly on GitHub
- Paper PDF opens in browser
- License is detected by GitHub

### 9 AM (US East / your local 6:30 PM if in India): post Twitter thread

Use `marketing/tweet_thread.md`. Replace placeholders:
- `[arxiv link]` → your arXiv URL
- `[github link]` → your GitHub URL
- `[zenodo doi]` → your Zenodo DOI

Post thread:
1. Compose tweet 1 with arXiv link
2. Reply to your own tweet for tweets 2-12
3. Pin the first tweet to your profile

Tag judiciously (only tweets 11-12):
`@AnthropicAI @OpenAI @GoogleDeepMind @MistralAI @cohere @huggingface @arxivlike`

### 11 AM: post on LinkedIn

Use `marketing/linkedin_post.md`. Personal post, not page.

### Throughout day: respond to every comment within 1 hour

Engagement signals matter for X's algorithm. Reply to every quote-tweet, comment, DM. Even simple "thanks for reading" replies. The first 3-6 hours determine whether the algorithm pushes the thread.

### Evening: monitor and amplify

- If thread gets >50 likes in 6 hours: it's working. Don't add to it; let it run.
- If it stalls under 20 likes: that's normal for first-time authors. Move to step 4 sooner.

---

## Day 4 — Hacker News submission

**Best time:** Tuesday or Wednesday, 8-10 AM US Eastern (lots of HN users browse at start of US workday).

Don't post on Friday/weekend (lower traffic, harder to make front page).

### Submission

1. Title: `Role-Indexed Attention: typed edges for binding in transformers`
2. URL: arXiv link (NOT GitHub — HN convention prefers the source)
3. **Within 30 seconds of submitting**, post the seed comment from `marketing/hn_post.md` as a reply to your own thread

### Engagement rules

- Respond within 15 minutes to every comment that gets >1 vote
- Be technical, never defensive
- If someone says "this won't work at scale": agree it's untested, link to what we measured
- If someone says "this is just X with extra steps": engage on the specific X, point to the difference
- If you go to front page: respond to every top-level comment for the next 6 hours

### What to expect

- Most arXiv-link submissions die at 0-3 votes (default front page floor is ~30)
- Front-page probability: ~10-25% with good title and a strong seed comment
- If front page: 30-100k views, 100+ stars on the GitHub repo, 5-20 emails from researchers

---

## Day 5–14 — Email outreach + sustained engagement

### Daily target: 3 cold emails per day (NOT bulk)

Use `marketing/cold_email.md`. Personalize each. Track in a spreadsheet:

| Date sent | Recipient | Lab | Status | Notes |
|---|---|---|---|---|

### Outreach schedule

**Day 5:** 3 senior researchers at Anthropic (Olah, Olsson, Bowman)
**Day 6:** 3 senior researchers at DeepMind (Hill, Shanahan, Blundell)
**Day 7:** 3 at OpenAI (research scientists in interpretability/architecture)
**Day 8:** 3 academics (Andreas at MIT, Linzen at NYU, Liang at Stanford)
**Day 9:** 3 at smaller labs (Hooker at Cohere, Biderman at Eleuther, Bubeck at MS)
**Day 10–14:** follow up on day-5 emails (single polite ping if no reply); send to next tier (Mistral, AI21, etc.)

### Email rules

- Subject lines that work: questions, specific paper references, never "Hi" or "Quick question"
- Send at 8-10 AM their local time (tools like Boomerang for Gmail can schedule)
- Length: under 200 words
- One ask: 10-min call, OR thoughts via email — never both
- Sign-off includes paper link, GitHub, your email — make follow-up frictionless

### Conversion expectation

Out of 30 cold emails:
- 5-10 read and ignore
- 5-10 read and reply with 1-line ack
- 3-5 reply substantively
- 1-2 lead to a real call
- 0-1 lead to a job conversation

Two real calls is enough to dramatically change your career trajectory.

---

## NeurIPS submission (separate timeline — May 2026)

NeurIPS 2026 deadlines (typical pattern):
- **Abstract registration:** mid-May
- **Full paper submission:** late May (one week later)
- **Reviews back:** late August
- **Author response:** September
- **Final decisions:** late September
- **Conference:** December

### What you submit

1. **Abstract** (250 words max) — same as paper abstract, trimmed
2. **Paper** — the anonymized `main_neurips.pdf`, max 9 pages main + unlimited refs/appendix
3. **Supplementary material** — code zip, all results jsonl, longer appendix
4. **Reproducibility checklist** — NeurIPS requires this; ~30 questions, all "yes" if your code is public
5. **Broader impact statement** — 1 paragraph on possible societal effects (mostly anodyne for this work)

### Critical: anonymization

Before submitting:
- Replace all `github.com/rishivhare/...` URLs with `[anonymized]` or `https://anonymous.4open.science/...` (free anonymizing service)
- Remove all author/affiliation info from PDF metadata: `pdftk main_neurips.pdf dump_data | grep Author` should show nothing
- Don't cite your own past work in a way that gives away identity

### Submission

1. Log into OpenReview.net
2. Find NeurIPS 2026 conference page when it opens (May)
3. Click "Submit"
4. Upload anonymized paper + supplementary
5. Add abstract, keywords, broader impact
6. Add author info (visible to OpenReview, hidden from reviewers via double-blind)

### Don'ts

- Don't tweet the paper between submission and reviews back. Posting on arXiv is fine and explicitly allowed; loud promotion can be considered un-anonymous behaviour.
- Don't share the OpenReview link until after the conference.

---

## Tracking

Create a spreadsheet (`results/launch_tracker.csv`):

```
date,channel,action,outcome
2026-04-30,Zenodo,Deposit,DOI:10.5281/zenodo.XXXXX
2026-05-01,arXiv,Submit,QUEUED
2026-05-02,arXiv,Live,arXiv:2605.XXXXX
2026-05-03,GitHub,Public,2 stars day 1
2026-05-03,Twitter,Thread,42 likes / 5 RT day 1
2026-05-04,HN,Submit,front page rank 28 / 4hr
2026-05-05,Email,Olah,sent
2026-05-12,Email,Olah,reply: "thanks, will read"
...
```

Review weekly. After 30 days, evaluate which channels worked.

---

## Failure modes and recoveries

**arXiv rejected (rare):** check moderation reason. Usually just a formatting issue (font embed, page numbers). Fix and resubmit.

**Tweet thread underperforms:** don't delete. Move to HN sooner. Or wait 2 weeks and re-thread with a different opening hook.

**HN goes to back page (~30 votes):** that's normal. ~5K views still. Move on.

**No email replies after 30 days:** check spam folder for replies. Re-evaluate cold-email templates with a friend. Try LinkedIn DMs to the same people.

**NeurIPS rejected:** submit to ICLR (deadline ~September) or ACL (Feb). Or workshop venues at NeurIPS itself (e.g., MATH-AI, MechInterp workshop).

---

## What success looks like at day 30

Realistic median outcome:
- Zenodo DOI live, ~50 downloads
- arXiv paper live, ~200 abstract views
- GitHub repo: 30-100 stars
- Twitter thread: 200-500 likes
- HN: didn't make front page
- Email outreach: 1-3 substantive replies, 0-1 calls
- Total time spent: ~80 hours

Success: 1 person at a notable lab read the paper carefully. That's the seed. Compounds over months.
