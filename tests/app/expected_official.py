"""Literal expected values for the official run (spec section 11), in one place.

When a new official run replaces the current one, update this file in one edit. Prefer deriving values from the run's
files in tests; use these only for numbers the spec fixes as acceptance values or that need hand-checked meaning.
Values marked (verified) were checked against the 7 Nov 18:20 run's files on 8 Nov.
"""

RUN_ID_PREFIX = "20261107-182015"           # official run id starts with this (verified)
SHOW_TITLE = "I Married a Broken Soldier"    # (verified)
RUN_LINE = "Official results · 50 episodes · scored by Nemotron 3.5 Super (free tier) · run 7 Nov 2026, 18:20"  # (verified)

# Promos (5.1)
N_PROMOS = 20               # (verified)
EPISODES = 50               # (verified)
EPISODES_USED = 17          # distinct segment episodes (verified)
SEGMENTS = 25               # (verified)
JOINS = 5                   # promos with 2 segments (verified)
FAIL_COUNT = 12             # promos failing any editor's check (verified)
TOP10_FAIL = 5              # (verified)
PARTS = (("opening", 1, 10, 11), ("middle", 11, 25, 7), ("late", 26, 50, 2))   # block counts verified
TO_CHECK_RANKS = (3, 4, 5, 6, 7, 9, 10, 12, 14, 18)
CHECKS_CELL = {3: "7 of 9", 10: "6 of 9"}   # (verified)
SEGMENTS_OF_5 = ["Ep 10 01:22.50–01:34.00", "Ep 11 00:00.00–00:20.00"]   # (verified)
RESOLVES_RANK = 10          # rubric.ends_open is false (verified)
CARD_HEADLINES = ("20 promos from 17 of 50 episodes", "Post episodes 1–10 first", "12 of 20 fail an editor's check")
BEST_CLIP_WITH_CLIP = 45
BEST_CLIP_PASS_ALL = 12
NO_CLIP_ALL_ZERO = (8, 17)
NO_CLIP_FLAT = (35,)
NO_CLIP_NO_FIT = (14, 25)

# Drawer (5.3)
DRAWER_3_HEADING = "#3 · 1 segment · 27.3 s"                 # (verified: 1 segment, 27.3 s)
DRAWER_3_CUT = "Ep 3  00:58.20–01:25.50  27.3 s"             # (verified)
DRAWER_3_BEFORE_POSTING = ("Before posting · 1", "Crosses a scene change")   # scene_change true (verified)
WHY_3_RIVALS = {4: ("0.809", "0.619"), 6: ("0.791", "0.605")}
PROMISE_9 = "Not verified: the AI named episode 3, which is before this clip, so code rejected the answer."

# Ranking (5.4)
WHY_19 = "Highest score of all 20 (0.945)"                   # #19 has the top score, 0.9452 (verified)
WHY_4 = "counted as 0.687"
WHY_12 = "(0.900)"
TAKEAWAY_1 = "11 of the 20 promos come from episodes 1–10"

# Quality (5.5)
PROBLEM_SNIPPETS = ("0.91", "1.06", "1.40", "1.19", "1.72", "40%", "43%", "52%", "12 of the 20", "14 of 50", "8 of the 24",
                    "1–7, 9–11, 24, 26–34", "43 of 50", "37 of 50", "4 to 1")
MEASURED_SNIPPETS = ("3.8 for the 20 promos (14 rated 4 or 5)", "3.65 over all 40 shortlisted clips",   # (verified)
                     "10 opening · 0 middle · 0 late")
CUES_HIDDEN_PCT, OFFICIAL_RECALL_PCT = "76%", "74%"          # (verified via the picker caption)

# Method (5.6)
METHOD_SNIPPETS = ("10 new answers, 29 reused", "4 refused", "158,057", "153,388")   # tokens verified
PAID_ESTIMATES = ("$0.43", "$0.31", "$1.54")
ASSUMPTION_ROWS = 13

# Lane R additions (8 Nov). The 12-drama benchmark, real reruns and weight checks replace the 2-drama claim in problem 1
# (sources: docs/research.md "The same benchmark on 12 dramas"; scripts/weight_sensitivity.py; scripts/weights_vs_editors.py).
N_PROBLEMS = 8              # the spec's 7 plus "weights set by hand"
PROBLEM_SNIPPETS_V2 = ("1.07×", "1.03×", "1.39×", "1.08×", "1.74×", "1.44×", "8 of 11 dramas", "68%", "88%", "8.75",
                       "p = 0.12", "40%", "43%", "52%", "12 of the 20", "14 of 50", "8 of the 24",
                       "1–7, 9–11, 24, 26–34", "43 of 50", "37 of 50", "4 to 1")
PROMISE_CASE_LINE = "Every question is answered except 1 that could not be verified."
CASE_LABELS = ("Fail an editor's check · 12", "Hard for a first-time viewer (AI) to follow · 6",
               "Edits to check on video · 5 joins, 3 scene changes", "Answers not confirmed · 2",
               "Episodes the AI answered flat · 14 (3 still flat)",
               "Best clip misses the writers' cliffhanger · 8 of 24 checked",
               "Does the show answer each promo's question? · 19 of 20")
WHY_3_FULL = ("Posted ahead of #4 (0.809) and #6 (0.791), which counted only 0.619 and 0.605 at this point (#4: a second "
              "promo from episode 7, ×0.85; opens on a humiliation like #2, ×0.90. #6: a second promo from episode 1, "
              "×0.85; opens on a humiliation like #2, ×0.90).")
SEPARATOR_LABELS = ("Opening · episodes 1–10<br>11 promos", "Middle · episodes 11–25<br>7 promos",
                    "Late · episodes 26–50<br>2 promos")
FLAT_NOTE_EPS = (8, 17)     # "AI scored every line 0, twice"

# Method columns and cost (5.6), Lane M. (verified against metrics.json, parse_report.json, aliases.yaml, promos.json)
METHOD_COLUMN_SNIPPETS = ("24 requests: 10 for the 50 episodes, 14 to re-ask flat answers", "8 requests, 5 clips each",
                          "3 answered; 2 refused by the daily limit", "3,727 blocks", "1,405 candidates",
                          "Runs 7 of the 9 editor's checks", "Shortlists 40", "11 / 7 / 2 places",
                          "1 in this run", "5 joins, 3 scene changes", "2 in this run", "8 name merges",
                          "5.6 min from start to finish")
CONSISTENCY_MINUTES = "6.9 and 7.3 min"     # stability_base 413.1 s, stability_rerun 439.7 s (verified)
LEARN_RECOVERY_POSTS = "500 posts"          # README and tests/test_learn.py: known weights recovered from 500 posts
