# What makes a good promo — research behind the editor's rubric

Promos are subjective, but the industry is not guessing. Ad platforms publish creative standards, micro-drama
studios test thousands of clips a week and their ad-intelligence vendors report what wins, trailer editors have a
craft vocabulary, and there is a small research literature on cutting trailers from screenplays. This note
condenses that into rules a **script-only** tool can check. Tags: **[E]** = the source says it; **[I]** = our inference.

## 1. How the people who do this for a living judge a promo

**Micro-drama studios (ReelShort, DramaBox, ShortMax, GoodShort, Kuku TV).**
- Volume and speed: ReelShort ran 400 to 1,000+ new creatives a day by mid-2023 [E]. The average short-drama app ran about 5,770 unique creatives a month in 2025 [E]. A clip burns out in 2 to 5 days, and Chinese editors cut 28 to 40 clips a day from episodes 1 to 20 [E]. *Implication: an editor wants many distinct, good-enough candidates fast, not one perfect clip.* [I]
  Sources: appgrowing.net/blog/en/reelshort · martechvibe.com/?p=34236 · m.huxiu.com/article/2451889.html
- Hook themes that win are identity or status reversal, contract or forced marriage, public humiliation followed by revenge, forbidden romance, and betrayal [E] (appgrowing.net, 500+ creatives, 2026).
- Formula: "conflict + reversal". Open mid-scene with no caption intro, and end on an unresolved reveal followed by "What happens next?" [E] (insightrackr ReelShort analysis; SocialPeta via mobidictum).
- Chinese craft vocabulary [E] (huxiu, toutiao, woshipm, cbndata):
  - 黄金三秒, "golden three seconds": the most intense moment goes first.
  - 钩子, "hook": an open question at the start and especially at the end.
  - 爽点, "satisfaction point": the catharsis.
  - 打脸, "face-slap": a person who belittled the hero is proven wrong.
  - 压弹簧 / 放弹簧, "compress / release the spring": humiliation builds, then a reversal releases it.
- The best-performing ad pattern shows the hero humiliated and **stops just before he strikes back** [E] (jiemian.com/article/10927795.html). Plain arguments with no emotional stake are the standard example of bad material [E].
- Fake hooks backfire. Out-of-context or fabricated ad hooks drew backlash and takedowns [E] (huxiu.com/article/4720486.html).

**Ad platforms.**
- Google's ABCD framework is Attention ("start big", mid-action), Branding, Connection (emotion, don't overload) and Direction. In Kantar's AI-scored study of about 11,000 ads, ABCD ads showed +30% predicted short-term sales [E].
- Meta/Nielsen: up to 47% of a video ad's value lands in the first 3 seconds, and up to 74% in the first 10 [E] (secondary source, digiday).
- Hook rate is 3-second views divided by impressions. Hold rate is ThruPlays divided by 3-second views. These are what buyers track [E].

**Trailer craft.**
- Structure: cold open, setup, escalation, a "button" ending, and a line of rising action [E] (Derek Lieu, via jonnyelwyn.co.uk).
- Establish character and stakes within 10 seconds. "Pose the question and leave the movie to answer it" (Malatesta). Never resolve [E].
- MrBeast's production guide says to front-load the first minute, make the stakes rise, and deliver on what the thumbnail promised, otherwise people "feel lied to" [E] (leaked memo).
- Netflix artwork research [E] (about.netflix.com/en/news/the-power-of-a-picture):
  - Members spend 1.8 seconds per title.
  - Emotional faces beat neutral ones.
  - Images with more than 3 people perform worse.
  - Villains perform well.

**Research on picking clips from scripts.**
- Papalampidi, Keller & Lapata label five *turning points* in screenplays (TRIPOD, 2019) and use them to cut trailers (2021). Picking by sentiment intensity alone chose intense but off-plot shots; structure plus intensity worked. Their automatic trailers contained about 40% more spoilers than the real ones [E] (arxiv.org/abs/1908.10328, arxiv.org/abs/2111.08774).
- **HIVE / DramaAD (EMNLP 2025 Industry)** is the closest prior art: LLM-cut ads for short dramas [E] (arxiv.org/abs/2507.02790).
  - Scenes are scored against a fixed trope rubric: confrontation, divorce and counter-attack score 3; twist and cliffhanger score 2.
  - The opening must be understandable without context, and the ending must be suspenseful.
  - Removing those boundary checks dropped hook rate from 0.71 to 0.28.
  - Ads are drawn only from the first 30% of episodes. Human editors still win.
- Wilmot & Keller (2020) show that suspense is best predicted by **uncertainty about the outcome**, not intensity [E].
- Clipping tools such as OpusClip, Vizard and AI-Youtube-Shorts-Generator score hook, flow, value and trend. Users' main complaints are that clips cut mid-thought, need 15 seconds of missing context, and favour energy over substance. Only about 3 in 10 clips are usable [E].

**Story psychology.**
- Suspense is a threat whose outcome is withheld. Curiosity is an outcome whose cause is withheld [E] (Brewer & Lichtenstein).
- Curiosity needs *some* prior knowledge: zero context gives zero curiosity [E] (Loewenstein 1994).
- Suspense needs someone to root for and a threat that is likely but uncertain [E] (Zillmann).
- Cliffhangers raise arousal and suspense but did **not** by themselves raise intent to keep watching in a lab study (N=133) [E] (Wirz et al. 2023). So the open ending is necessary but not sufficient: the clip also needs investment in a person.
- Spoiler studies are mixed and only test *ending* spoilers [E]. Industry practice is not to cut from the finale [E].

## 2. Good promo vs bad promo

| A good promo… | A bad promo… |
|---|---|
| opens mid-conflict, with the most intense line or a physical shock in the first 3 seconds | opens on a greeting, scenery, exposition or recap |
| makes sense to a stranger: 2–3 people, one "X wants Y, but Z" | needs backstory: names, places or past events the viewer never saw |
| gives the viewer someone to root for (a wronged underdog) | has nobody to side with |
| escalates, so the stakes are higher at the end than in the middle | peaks early and sags |
| cuts **before** the payoff: the threat, question, reveal or slap is still open | spends its own payoff: apology accepted, rescue done, question answered |
| is faithful: it implies only events that really happen | stitches unrelated moments into a story the show doesn't tell |
| comes from early, spoiler-free episodes | gives away the endgame twist |
| works on mute: short, captionable lines plus visible action | rests on long monologues or off-screen voice |

## 3. Which scripts are easy or hard to promo

| Scene type | Easy or hard | Why |
|---|---|---|
| Public humiliation, cut before the payback | easy | a rootable underdog plus an unresolved threat [Zillmann; jiemian] |
| Slap, confrontation, threat, ultimatum, divorce or contract demand | easy | conflict and stakes are visible in one line |
| Identity or status reveal, betrayal discovered | easy | a power reversal; works as a "turn" |
| Hidden identity the viewer knows but the villain doesn't | easy | dramatic irony set up inside the clip (Hitchcock's bomb) [I] |
| Exposition, recap, scene-setting | hard | no gap and no emotion |
| Quiet romance or declaration | hard | low arousal; it is a *resolution*, not a cliff |
| Lines that lean on backstory ("after what you did that night") | hard | the stranger has no prior to be curious from [Loewenstein] |
| 4+ speakers, ensemble scene | hard | beyond what a stranger can track [Netflix; Cowan] |
| Payoff that is purely visual (a look, an object) | hard for a script-only tool | the page can't show whether it reads on screen |

| Show type | Easy or hard |
|---|---|
| Revenge, hidden heir, contract marriage, CEO/duke romance with two clear leads and simple moral sides | easy: many compress-and-release beats |
| Slow-burn romance, mystery or whodunit (every reveal is a spoiler and the hook needs context), large ensembles, voice-over-heavy or dialogue-light shows, morally grey leads | hard |

## 4. How this became code (`promo_finder/rubric.py`, `promoability.py`)

- **Per-line scores.** The LLM now also gives each line a *role* in the compress-and-release rhythm (setup, escalation, turn, payoff) and an *intensity*. The prompt is a micro-drama ad rubric (HIVE-style trope list, golden 3 seconds).
- **Editor's checks.** These are named checks run on every candidate clip. Each failed check multiplies the score by a fixed penalty set in `config.yaml`, and every verdict is shown on the clip's card.
  - Code checks: opens on conflict · opening needs no backstory · ≤3 speakers · ends before the payoff · builds to the end · no dead air.
  - Judge checks, run by the cold-viewer pass: stakes clear to a stranger · someone to root for · the clip doesn't answer its own question.
- **The slate.** Where promos come from is copied from professional practice (section 5): about 55% of the slate from the first 20% of episodes, 33% from 20-50%, 12% from the second half, opening posted first. `--paywall N` turns the opening into the free episodes.
- **Faithful by construction.** Every clip is a verbatim span. The only joins are tail-to-head across a direct continuation, or a hook shot followed by a later run in the same episode, and the latter is flagged.
- **Promo-ability report.** For every episode it gives a tier (easy, medium or hard) and the reasons: conflict density, payoff share, backstory load, cast size and off-screen share. A show-level verdict says whether the show is led by its two main characters or is an ensemble.
- **Blind A/B.** The new top 10 is compared against the previous version's top 10, slot by slot. A first-time-viewer judge sees each pair in both orders, and a side only wins if it wins both ways.

**What we still can't know from text:** whether a cliffhanger actually lifts installs (Wirz et al. say not on its own), and how acting and camera land. The honest next step is to log hook rate and hold rate per posted promo, then refit the weights and penalties on that data.

## 5. Measured on our own and external data (reproducible)

**Where professional editors take ad footage from** (`scripts/benchmark_dramaad.py stats`, DramaAD: 501 ads cut by professional editors from 28 Chinese short dramas of 72-109 episodes, https://huggingface.co/datasets/wonderful9462/DramaAD):

| Part of the show | Share of ad footage (seconds) | Share of ad segments |
|---|---|---|
| Opening: first 20% of episodes | 55% | 53% |
| Middle: 20-50% | 33% | 34% |
| Second half | 13% | 12% |

39% of all ad segments come from the first 10% of episodes, and 55% of ads open on footage from the first 20%. But no drama keeps all of its ads inside its first 30%: editors also reach for mid-show peaks. Ads there run a median of about 3 minutes, built from about 3 segments. Our promos follow the brief's 20-60 s instead.

**A trained model from Hugging Face.** No model on the Hub is trained to pick promos or trailers from scripts. Searches for highlight, trailer, suspense, cliffhanger and turning point found only a Twitch-chat highlight model and an undocumented suspense classifier. The closest useful one is `j-hartmann/emotion-english-distilroberta-base` (about 390K downloads; its 83 MB ONNX int8 version runs on a CPU at about 450 lines a second). Tested on all 1,859 lines of the pack:
- It agreed with the model's line intensity less than our keyword list did (Spearman 0.20 vs 0.32).
- Added to the keyword hook score, it did not significantly improve detection of strong hooks (AUC 0.676 to 0.683; bootstrap 95% CI of the gain -0.02 to +0.03).

Not adopted. A promo-worthy line is not the same as an emotional one: a quiet ultimatum reads as "neutral" to a classifier trained on tweets.

**Open-source models, unlimited and local** (Apple MLX on a laptop, scored on episodes 1-3 against the official model):

| Model | Same hook and cliff score as the official model | Within 1 point | Time per episode |
|---|---|---|---|
| Qwen3-4B-Instruct (4-bit) | 5% | 26% | about 60 s |
| Llama-3.1-8B-Instruct (4-bit) | 52% | 65% | about 79 s |
| (official model vs itself, re-run) | 61% | 95% | about 10 s per call, hosted |

Local models are unlimited and private, but a laptop takes about an hour per show, so they are the offline option, not the interview one.

**Free hosting, checked against the providers' own docs (November 2026).**
- **Streamlit Community Cloud:** free, about 1 GB guaranteed memory (0.7-2.7 GB), up to 2 CPUs, no GPU, sleeps after 12 hours without visitors.
- **Hugging Face Spaces:** Gradio and Docker Spaces, and therefore Streamlit, now need a paid plan to create. Only static and ZeroGPU Gradio Spaces stay free.
- **Hugging Face Inference Providers:** no free monthly credits since October 2026.

So the deployed app runs on Streamlit Community Cloud and calls a hosted model through an API key: by default a free open-weights model, NVIDIA Nemotron 3.5 Super on OpenRouter.

**Consistency** (`promo-finder stability`). Measured from two independent runs (1,294 lines), the chance a score comes back unchanged is 56-93% depending on the score. Redrawing every score from that noise and re-ranking 60 times gives:

| Ranking | Top 10 shared by two runs | Top 20 shared | A typical promo is kept in |
|---|---|---|---|
| Best scores anywhere (first version) | 37% | 42% | 45% of re-runs |
| Slate, one reading per episode | 40% | 43% | 57% |
| Slate, two readings averaged (`--votes 2`) | 52% | 54% | 63% |
| Slate, three readings averaged | 52% | 58% | 70% |

**External benchmark** (`scripts/benchmark_dramaad.py score`). Two unseen Chinese dramas, the first 20 episodes each, top 10 picks, scored against the ads their professional editors cut. *Usage* means how many of the drama's ads contain a given second.

| Strategy | Usage lift vs an average second | Clips mostly inside an ad | Lift within the same episode |
|---|---|---|---|
| Our slate | 1.40 | 45% | 0.91 |
| Our raw best moments | 1.21 | 55% | 1.14 |
| Random clips | 1.19 | 54% | 1.06 |
| Opening seconds of each episode | 1.72 | 65% | 0.89 |

The two dramas disagree (slate 2.0× on one, 0.75× on the other). Editors in DramaAD mostly reuse episode openings inside long (~3 min) multi-segment ads, so position predicts their footage better than moment-level taste does. This is the main open question; calibrating on all 28 dramas is the next step.

**Blind A/B with a judge from another model family** (Llama-3.1-8B, local). The first version's top 10 against the slate's: first version 4, slate 1, 5 ties, with the judge position-consistent on half the pairs. The slate posts early-episode clips the way professional editors do, and those are less explosive in isolation. That is a real trade-off, and `slate.shares` is the knob.

**Controlled test of the editor's checks** (`scripts/check_experiment.py`). The same judge rated clips that fail a check against clips matched on base quality that pass it (5+5 each). Mean "follows" score, pass vs fail:
- stops before the payoff: 5.0 vs 4.8
- no backstory: 4.6 vs 4.2
- ≤3 speakers: 3.6 vs 3.8, not confirmed

The test is small and indicative only.
