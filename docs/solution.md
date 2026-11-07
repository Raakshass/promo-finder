---
title: "promo-finder — Kuku AI Generalist Assignment, Problem 1"
---

# promo-finder

<p class="sub">Reads a micro-drama's timestamped scripts and hands the editor ready-to-cut promos: exact in and out points, the hook, the cliffhanger, why it works, which part of the show to take them from, and the question each one leaves the viewer with. Then it checks its own work.</p>
<p class="byline">Siddhant Jain · 23113142 · Problem 1: Find the Promos · Prototype {{PROTOTYPE_URL}} · Code {{REPO_URL}} · Raw outputs {{OUTPUTS_URL}} · Video {{VIDEO_URL}}</p>

**Why this problem:** editors watch every episode to find promo moments. This was the only problem with real data (50 timestamped scripts), so the tool can be checked against the writers' cliffhangers and professional editors' ads, then rerun on the next show with one command.

## What makes a promo work
"Good" sounds subjective. But the people who cut promos at scale agree on five things, and each one is a named check in the tool (sources in `docs/research.md`).

| A promo that works… | Evidence | How the tool checks it |
|---|---|---|
| **grabs attention within 3 seconds** | Meta: up to 47% of an ad's value lands in 3 s. Editors call it the "golden three seconds". | the hook starts ≤3 s after the cut, on conflict |
| **makes sense to a stranger** | Curiosity needs a visible gap (Loewenstein). HIVE/DramaAD, EMNLP 2025. | ≤3 speakers, no unexplained names, stakes clear to a first-time viewer |
| **gives you someone to root for** | Suspense needs a liked person in danger (Zillmann). | the first-time viewer names whom they side with |
| **stops before the payoff** | Micro-drama builds pressure, then releases it. Top ads cut just before the strike-back. | no release after the last escalation |
| **keeps its promise** | Fake-hook ads drew backlash. MrBeast: viewers who "feel lied to" leave. | verbatim clips; a ledger finds where the show answers the question |

## How it works
DIAGRAM_HERE

- **Where promos come from is measured, not guessed.** In DramaAD's 501 ads cut by professional editors, {{PRO_SHAPE}}. So the slate copies that shape and posts the opening first. `--paywall N` makes the opening the free episodes.
- **It reads the whole show.** Each episode is scored in full. One more AI call over all the episode summaries finds the leads, the turning points and the reveals that must not be spoiled.
- **Ranking is plain arithmetic in code, and the AI never writes a timestamp.**
    - score = (0.35·hook + 0.30·cliff + 0.25·stands-alone + 0.10·pace) × a fixed penalty per failed check × 1.05 for a turning-point episode.
    - Overlapping clips are dropped.
    - The slate fills its opening / middle / late places best-first, at most 2 per episode in the top 10.
    - The weights are hand-set priors in `config.yaml`, and every card shows its own arithmetic.

## Results on the real pack
{{TOP5}}

| Check | Result |
|---|---|
| Every clip follows the rules (20–60 s, hook ≤ 3 s, starts on a cut) | {{INVARIANTS}} |
| Passes all nine editor's checks · comes from the opening of the show | {{RUBRIC}} · {{SLATE}} |
| A first-time viewer follows the clip (1–5) | {{CV_MEAN}}; {{CV_GE4}} rate it 4 or 5 |
| Promise & payoff: the show answers the promo's question | {{LEDGER}} |
| **External benchmark:** {{BENCH_N}} unseen Chinese dramas, scored against professional editors' ads | {{BENCH}} |
| Blind A/B against my first version, judged by a different model family | {{AB}} |
| Consistency: two runs agree on… | {{CONSISTENCY}} |

## Assumptions, and how each was tested
{{ASSUMPTIONS}}

## Where it fails
{{FAILURES}} The first-time viewer is a model reading a transcript, not a person watching. The real proof is hook rate and hold rate on posted promos.

## Scale, cost, models, deployment
- **Cost and time.** {{CALLS}} calls per 50-episode show, **$0**: the official results come from a free open-weights model (NVIDIA Nemotron 3.5 Super, via OpenRouter's free tier).
    - Measured on that free tier: 7–17 minutes in our runs, held back by request caps. One episode is about 10 s of model time.
    - On an uncapped API, `--fast` runs 8 episodes at a time. The estimate is about 2 minutes per show; this has not been timed.
- **Swapping the model.** One setting: `PROMO_MODEL=provider/model`, or `custom/<model>` for any OpenAI-compatible endpoint, such as Kuku's own.
    - A local Llama 3.1 8B matched 52% of the line scores on 3 episodes, at about 80 s per episode on a laptop.
- **Free deployment.** The app runs on Streamlit Community Cloud and opens on the official results.
    - Re-runs on a new show use the same open model, through NVIDIA's free API (no daily cap, about 40 requests a minute) when the deployment has its key, otherwise through OpenRouter's free tier (about 50 calls a day per key).
    - Visitors can also use their own key, or no key.
    - The model's weights are open, but at 120B parameters (75 GB in 4-bit) no free host can serve them; NVIDIA's free API already does.

**Paid upgrades, and the free version built for each:**

- **A frontier model read twice** (`--votes 2`); free version: the open model, where two readings raise simulated top-10 agreement {{VOTES_GAIN}}.
- **Video understanding and speech alignment;** free version: script timing plus `cut_promos.sh` (ffmpeg).
- **Feedback from the Meta / TikTok ads APIs;** free version: an approve / reject log to refit the weights.

## AI tools, and where they were wrong
An AI coding agent built this with me, and every claim was rechecked with code. It got five things wrong, each logged with its fix in `docs/ai_log.md`. It invented a count (25 "cliffhanger" notes, where a grep finds 12). It read a spending limit as a balance. Its first version never judged a whole clip. It trusted the free model, though 14 of 50 answers came back flat. And it put a script line into a public prompt.
