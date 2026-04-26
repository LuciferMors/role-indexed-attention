# Launch checklist (the only file you need open)

Print this. Tick boxes as you go.

## Before launch (1 day prior)

- [ ] All tests pass: `make test`
- [ ] Paper PDFs final (`main_zenodo.pdf` named, `main_neurips.pdf` anonymous)
- [ ] README polished and accurate
- [ ] LICENSE file present
- [ ] `.gitignore` excludes secrets and large logs
- [ ] Twitter profile filled (bio mentions ML research)
- [ ] Personal website live with paper link
- [ ] Email signature updated with paper link
- [ ] Cold-email tracker spreadsheet created (`results/launch_tracker.csv`)

## Day 1 — Zenodo

- [ ] Sign up at https://zenodo.org/signup
- [ ] New upload: type=Preprint
- [ ] Upload: main_zenodo.pdf, main_zenodo.tex, refs.bib, code.zip
- [ ] Author: Rishi Shivhare, ORCID (create at https://orcid.org if needed)
- [ ] License: MIT (code), CC-BY-4.0 (paper)
- [ ] Keywords: attention, transformers, compositional generalization, frame semantics
- [ ] **Publish** — record DOI: ____________________

## Day 2 — arXiv

- [ ] Submit at https://arxiv.org/submit
- [ ] Primary: cs.LG · Secondary: cs.CL, cs.AI
- [ ] Upload .tar.gz of LaTeX source
- [ ] Comments: include GitHub link and Zenodo DOI
- [ ] Submit; wait 1-2 days for moderation
- [ ] Record arXiv ID when live: ____________________

## Day 3 — GitHub + Twitter + LinkedIn (single morning)

- [ ] Make GitHub repo public: ____________________
- [ ] Verify README renders, paper PDF opens
- [ ] Post Twitter thread (use marketing/tweet_thread.md, fill in links)
- [ ] Pin first tweet
- [ ] Post on LinkedIn (use marketing/linkedin_post.md)
- [ ] Reply to every comment within 1 hour for first 6 hours

## Day 4 — Hacker News (Tuesday/Wednesday 8-10 AM US Eastern)

- [ ] Submit at https://news.ycombinator.com/submit
- [ ] URL: arXiv link
- [ ] Title: "Role-Indexed Attention: typed edges for binding in transformers"
- [ ] Within 30 seconds, post seed comment from marketing/hn_post.md
- [ ] Monitor for 6 hours; reply to every comment
- [ ] Result: ____________________

## Day 5 — First batch of cold emails

Use marketing/specific_emails.md. Paste, personalize bracketed lines, send.

- [ ] Chris Olah (Anthropic) — sent at ___:___
- [ ] Sam Bowman (Anthropic / NYU) — sent at ___:___
- [ ] [One more from your list] — sent at ___:___

## Days 6-14 — Sustain

3 emails/day. Track replies in spreadsheet.

- [ ] Day 6: 3 sent
- [ ] Day 7: 3 sent
- [ ] Day 8: 3 sent
- [ ] Day 9: 3 sent
- [ ] Day 10: 3 sent + follow-up day-5 emails
- [ ] Day 11: 3 sent
- [ ] Day 12: 3 sent
- [ ] Day 13: 3 sent
- [ ] Day 14: review tracker, plan month 2

## NeurIPS submission (separate, May 2026)

- [ ] OpenReview account created
- [ ] Run `bash marketing/anonymize_for_neurips.sh`
- [ ] Manually verify anonymized PDF: NO author info anywhere
- [ ] Strip PDF metadata (exiftool)
- [ ] Submit at NeurIPS 2026 OpenReview page when open
- [ ] Submit abstract by abstract-deadline (typically 1 week before paper)
- [ ] Submit full paper + supplementary
- [ ] Complete reproducibility checklist (all "yes" — code is public)
- [ ] Add broader impact statement
- [ ] Confirm submission received

## Daily during NeurIPS submission week

- [ ] Don't tweet about the paper publicly
- [ ] Don't share OpenReview link
- [ ] Continue cold-email outreach (those are not anonymous)

## After NeurIPS reviews back (August)

- [ ] Read all reviews carefully, don't react emotionally
- [ ] Write rebuttal in author response window (typically 1 week)
- [ ] Address every reviewer concern with either a fix or a clear explanation
- [ ] If accepted: prepare camera-ready, plan conference attendance
- [ ] If rejected: revise based on feedback, submit to ICLR (~Sept) or ACL (~Feb)

---

## Emergency contacts / what to do if stuck

- arXiv submission stuck in moderation > 4 days: email arxiv-help@cornell.edu politely
- Zenodo question: https://zenodo.org/support
- NeurIPS question: program-chairs@neurips.cc (only after reading FAQ)
- Twitter thread crashes: don't delete; let it rest; re-engage with a reply
  in 7 days

---

## What to celebrate at each milestone

- Zenodo DOI live: small. Permanent record exists.
- arXiv live: medium. Citable, indexed.
- GitHub repo public + ≥10 stars: small. Visibility.
- HN front page: large. Real audience.
- 1 substantive email reply: large. Real connection.
- 1 phone call with a researcher: very large. Begin career.
- NeurIPS accept: very large. Career-defining single artifact.

Don't celebrate the paper itself until it's accepted. Until then, you have a
preprint with promise. The work continues.
