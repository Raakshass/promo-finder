---
title: "promo-finder — Kuku AI Generalist Assignment, Problem 1"
---

# promo-finder

<p class="sub">It reads a micro-drama's timestamped scripts and gives the editor promos that are ready to cut: exact in and out points, the hook, the cliffhanger, one line on why each works, and which to post first. Then it checks its own picks and says where they are weak.</p>
<p class="byline">Siddhant Jain · 23113142 · Problem 1: Find the Promos · App {{PROTOTYPE_URL}} · Code {{REPO_URL}} · Raw outputs {{OUTPUTS_URL}} · Video {{VIDEO_URL}}</p>

## The problem, and why I picked it
Today an editor watches every episode to find promo moments, and a new show ships every week. This was the only problem that came with real data (50 timestamped scripts), so I could check the tool against the writers' own cliffhangers and against ads cut by professional editors, then rerun it on the next show with one command.

## What makes a promo work
"Good" sounds subjective, but the people who cut promos at scale agree on five things. Each one became a check in the tool (sources in `docs/research.md`).

| A promo that works… | Evidence | How the tool checks it |
|---|---|---|
| grabs attention in 3 seconds | Meta: up to 47% of an ad's value lands in the first 3 s | the hook line starts within 3 s of the cut, on conflict |
| makes sense to a stranger | curiosity needs a visible gap (Loewenstein); HIVE/DramaAD, EMNLP 2025 | at most 3 speakers, no unexplained names, clear stakes |
| gives you someone to root for | suspense needs a liked person in danger (Zillmann) | a first-time viewer names whom they side with |
| stops before the payoff | the best micro-drama ads cut just before the strike-back | nothing is resolved after the last escalation |
| keeps its promise | fake-hook ads drew backlash | clips are verbatim, and the tool finds where the show answers the question |

## How it works
DIAGRAM_HERE

The AI scores every line 0–3 for hook (would a stranger stop?) and cliff (would cutting here leave them hanging?). Code builds every clip that starts on a camera cut, has its hook in the first 3 s, ends just after a cliff line and runs 20–60 s, then ranks them. The "first-time viewer" in step 4 is a model that sees only the clip, never the show. The editor makes the final call in the app and exports a cut list or an ffmpeg script.

## How a promo is ranked
<div class="formula">score = (0.35 × hook + 0.30 × cliff + 0.25 × stands alone + 0.10 × pace) × the penalty of each failed check × 1.05 for a turning-point episode × 0.9 for a two-part clip</div>

- **hook**: the model's 0–3 score for the clip's opening line, divided by 3. A physical action in the first 3 s adds half a point first.
- **cliff**: the same for the closing line. A writers' cue at the cut adds half a point; a love line that is not cut off counts at most 2 of 3.
- **stands alone**: for the 40 shortlisted clips, the first-time viewer's rating out of 5. Before that, an estimate from the unexplained names in each line.
- **pace**: the share of the clip that is spoken (full marks at half), times a length factor of 1 for 25–45 s, falling to 0.6 at 20 or 60 s.
- **penalties**: each failed check multiplies the score by 0.75 (gives away the payoff) up to 0.92.

**Why these weights.** In a feed, a promo has to win three things in order: the viewer stops, follows, then taps. Hook comes first because a viewer who does not stop never sees the rest. Cliff comes next because the brief defines a promo by its hook and its cliffhanger. Stands-alone is third, but four of the nine checks test it too. Pace is last because a script is a weak stand-in for the edit's rhythm. These are hand-set starting values, not fitted ones, so I tested them three ways:

- **Small changes barely matter.** Moving any one weight by 0.05 keeps about 9 of the top 10 (8.75 on average, 9.25 if another cut of the same moment counts). The split between opening, middle and late promos never moves.
- **Professional editors do not contradict them.** On two dramas with professional ads, fitting the four terms to the clips editors used could not reject these proportions (p = 0.12), and weights tuned on one drama did not carry over to the other. Only pace showed a lead, and it did not hold inside episodes.
- **Real traction should set them.** `promo-finder learn` refits the weights from posted promos' hook rate, hold rate or the editor's approvals. It counts the current weights as 30 posts' worth of evidence, so a handful of posts cannot swing them, and `--plan K` picks the next posts by Thompson sampling, so uncertain clips still get a turn. On simulated data it recovers known weights from 500 posts (median worst error 0.026). It has not seen real posts yet.

## Results on the real pack
{{TOP5}}

| Check | Result |
|---|---|
| Follows the brief's rules: 20–60 s, hook in the first 3 s | all 20 (21.9–56.3 s; every hook within 1.5 s of the cut) |
| Where the promos come from | 17 of the 50 episodes; 5 promos join two episodes; 11 from episodes 1–10, 7 from 11–25, 2 from 26–50 |
| A first-time viewer (a model) follows the clip, 1–5 | 3.8 on average; 14 of the 20 rated 4 or 5 |
| Passes all nine editor's checks | 8 of the 20 (5 of the top 10); the others are ranked lower, not hidden |
| The show answers the promo's question | 19 of 20 (11 later in the same episode, 8 in a later one); the 20th could not be verified |
| Professional ads on 11 public dramas (DramaAD) | Over the top 20, professional ads use our seconds 1.39× as often as an average second (random clips 1.08×). Inside an episode, the best-scored clips beat random spots by a small margin (+0.04, 95% CI 0.00 to 0.09). Simply taking each episode's opening still matches the editors better in the top 10 (1.74× against our 1.44×). |
| Two uncached runs agree on… | 68% of the top-10 candidates and 88% of hook scores on 4 public dramas. On Kuku's show, simulated from measured noise: 40% of the top 10, 52% with `--votes 2`. |

## Where it picks badly
- **It reads text, not video.** Acting, faces and the real edit's timing are invisible, and it assumes script time is cut time. `cut_promos.sh` renders every clip so a person can check it.
- **Its judges are models.** The first-time viewer reads a transcript; it is not a person watching.
- **A blind comparison went against the new version.** A judge from another model family preferred my first version's top 10, 4 to 1 with 5 ties. The new version posts early-episode clips the way professional editors do, and those are less explosive on their own. How many promos come from the opening is one setting.
- **The free model is uneven.** It came back flat on 14 of the 50 episodes; asking again fixed 11.
- **Runs differ.** More than half of the top 10 can change between runs on Kuku's show; reading each episode twice helps.

## Assumptions, and how each was tested
| Assumption | How it was tested | Result |
|---|---|---|
| Promos come mostly from the opening | 501 professional ads (DramaAD) | 55% / 33% / 13% of their footage comes from the first 20% / next 30% / second half of a show; the posting order copies this |
| An end-of-episode + start-of-next join is one moment | the writers' continuity notes | 44 of 50 say the next episode picks up directly |
| The model reads the drama, not the stage directions | the same run with "cliffhanger" and "cut to black" hidden | it finds the known cliffhangers 76% of the time, against 74% with them |

All 14 assumptions, with what to change if each is wrong, are in `docs/assumptions.md`.

## Scale, cost and models
- **Cost and time.** 43 model calls per 50-episode show, $0 on free tiers. The official run used NVIDIA's open Nemotron 3.5 Super through OpenRouter's free tier (7–17 minutes, held back by request caps). On NVIDIA's free API, `--fast` ran 20 episodes in 2.4 minutes, so a full show should take about 6 (an estimate).
- **Swapping the model.** One setting: `PROMO_MODEL=provider/model`, or `custom/<model>` for any OpenAI-compatible endpoint, such as Kuku's own. With no key, keyword rules run the same pipeline.
- **Paid upgrades, each with a free version built:** a frontier model reading each episode twice (free: `--votes 2`), video understanding (free: an ffmpeg render of each clip), and ad metrics feeding `learn` (free: the editor's approve and reject log, which also keeps a person in the loop).

## AI tools, and one place they were wrong
I built this with an AI coding agent, and the tool itself uses open models: Nemotron 3.5 Super scores and judges, and Llama 3.1 8B served as an independent judge. One place the agent was wrong: it reported that 25 episodes carry a "cliffhanger" note, but a search of the scripts finds 12. Since then, every number in this doc comes from a script, not from the agent's summary.
