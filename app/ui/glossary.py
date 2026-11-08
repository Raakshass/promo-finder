"""The one source of wording: glossary (spec 7), the editor's checks (spec 4.2), word maps and decision labels.

GLOSSARY and CHECKS are copied from the spec tables. Values in braces, such as {leads} in a check rule, are filled by
the caller with str.format. Penalties are never written into CHECKS: read them from the run's config with penalty().
GLOSSARY holds no number that belongs to one run (an episode range, a place count, a candidate count), because it is
shown for every run. The weights, penalties and cut-offs it keeps are definitions taken from config.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    key: str
    pass_label: str
    fail_label: str
    rule: str
    decided_by: str


GLOSSARY: dict[str, str] = {
    'Promo':
        'A 20–60 s clip from the show, cut to pull new viewers in: one stretch, or two joined.',
    'Segment':
        'One continuous stretch of one episode inside a promo. A promo plays its segments in order.',
    'Episode time':
        'Minutes:seconds.hundredths from the start of the episode, as written in the script, e.g. 01:22.50.',
    'Clip time':
        'Seconds from the start of the promo, e.g. "starts 0.1 s in".',
    'Show time':
        'Minutes of watching from the start of episode 1, e.g. "13.2 min into the show".',
    'Hook':
        'The opening line or action. It must start within the first 3 s of the clip.',
    'Cliffhanger':
        'The last line or action before the cut: the moment left unresolved.',
    'Closing line':
        'Shown instead of "Cliffhanger" when the clip answers its own question before the cut.',
    'Why it works':
        'One line from the first-time viewer (AI) on why the clip works. AI-written; it can be wrong.',
    'AI line score (0–3)':
        "The AI's rating of one line. Hook: 3 = a slap, a shock, a jaw-dropping reveal or a line that needs no context; 2 = a strong confrontation; 1 = mildly interesting; 0 = filler. Cliffhanger: 3 = an ultimatum, a reveal mid-reaction, a threat or a question left hanging; 2 = clearly unresolved; 1 = mild; 0 = resolved. Only lines rated 2 or 3 can open or end a promo.",
    'Posting order (#)':
        'The order to post the promos in: post from the top. Not a quality rank.',
    'Part of the show':
        'Opening (the first fifth of the episodes), middle (to the halfway point) or late (the second half). Each part gets a fixed share of the places.',
    'Score':
        '0 to 1: 0.35 × hook + 0.30 × cliffhanger + 0.25 × makes sense alone + 0.10 × pace, × a penalty per failed check, × 1.05 for a turning-point episode.',
    'Counted as':
        'The score after the variety rule, at the moment the promo was placed.',
    'Variety rule':
        "Inside each part: ×0.85 for each promo already taken from the same episode, ×0.90 for opening on the same kind of moment as the promo before, and at most 2 per episode in a part's first 10 places.",
    'Makes sense alone':
        'The first-time viewer\'s (AI) "follows it" rating ÷ 5. Before that test, a text estimate from unexplained names.',
    'Pace':
        'How much of the clip has someone speaking (full marks from 50%, halved below 35%) × a length factor that is best at 25–45 s.',
    "Editor's checks":
        'Nine named rules from how promo editors work. A failed check lowers the score by a fixed factor; nothing is removed.',
    'Before posting':
        'Things to check on the video or in the story before posting: joins, scene changes, twist episodes, spoilers, unconfirmed answers, clips that end after their payoff.',
    'Weak spots':
        'Known weaknesses already priced into the score: failed checks, a low first-time-viewer rating, a fragile pick.',
    'First-time viewer (AI)':
        "The same AI model reads only one clip's spoken lines and actions and answers as a stranger scrolling past. A stand-in, not a person.",
    'Follows it (1–5)':
        '5 = fully follows who wants what; 3 = gets the gist; 1 = lost.',
    'Would tap (1–5)':
        'Would it tap to watch the show? Shown for context; not part of the score.',
    "Couldn't place":
        'Names or things the first-time viewer (AI) did not understand from the clip alone.',
    'Shortlist':
        'The best clips after overlaps are dropped. Only these get the first-time-viewer test; the promos are chosen from them.',
    'Candidate clips':
        'Every clip that fits the rules, before overlaps are dropped.',
    'Promise and payoff':
        'The question a promo leaves, and where the show answers it. A question never answered makes a fake hook.',
    'Wait after the clip':
        "Minutes of watching from the clip's end to the start of the episode that answers its question.",
    'Confidence':
        'High: a summary states it. Medium: clearly implied. Low: the AI is guessing.',
    'Not verified':
        'The AI named an impossible episode (before the clip, or after the last one), so code discarded the answer.',
    'Turning point':
        "One of five story turns the AI found in the writers' notes. Clips from those episodes get ×1.05, except the climax.",
    'Major reveal':
        "A twist the AI found in the writers' notes. Clips from that episode are flagged so they don't give it away.",
    'Spoiler risk':
        'The clip comes from the last fifth of the show.',
    "Writers' cue":
        'A stage direction such as "cliffhanger" or "cut to black" within 2 s of the cut. It adds 0.5 to the 0–3 cliffhanger score.',
    'Physical hook':
        'An action (a slap, a grab, a fall) lands in the first 3 s. It adds 0.5 to the 0–3 hook score.',
    'Join':
        'The end of one episode followed by the start of the next, which picks up the same moment. Check that it plays.',
    'Scene change':
        'A new scene starts inside the clip.',
    'Steady / usually picked / fragile pick':
        "Kept on the shortlist in at least 80% / 50–79% / under 50% of re-runs simulated from the model's measured noise. The cut-offs are this app's display choice.",
    'Flat answer':
        'The AI rated no line above 1 as a hook, or none above 1 as a cliffhanger, for a whole episode: usually a lapse. The episode is asked again alone.',
    'Reused answer':
        "An identical earlier request's answer, served from the cache at no cost.",
    'Refused request':
        "A request the free tier's daily limit turned down.",
    'Keyword rules':
        'The no-AI baseline: a word list scores each line (slaps, threats, a question at the end). Instant and free.',
    'Lift':
        "How often professional editors' ads used the chosen seconds, against an average second of the evaluated episodes (1.0 = average). Within-episode lift compares against the same episode.",
    'Blind A/B':
        'A judge model sees two clips per slot without knowing which version made them, in both orders; a side wins only if picked both times.',
    'Difficulty':
        'Easy: a clip with a strong hook and cliffhanger (2+ of 3) passes all 7 script checks. Medium: a strong clip opens on conflict and stops before the payoff but fails a softer check. Hard: no strong clip does both. Not scored: the AI answered the episode flat.',
    'Official results / Evidence run / Your run':
        'Official: the run the solution doc reports. Evidence: a run kept to back one measured claim; read-only. Your run: a run started in Run it, kept until the app restarts.',
    'AI pitch':
        'A one-line pitch the AI wrote for each top-10 promo after code set the order. Not the reason for the position.',
    'Read-along':
        "The clip's lines played at their script timing. Text only; there is no video.",
    'Your call':
        'Approve, Trim or Reject. Kept in this browser tab; download your review to keep it.',
    'Cut list':
        'One row per segment to cut, with your decisions.',
}

CHECKS: dict[str, Check] = {c.key: c for c in (
    Check('opens_on_conflict', 'Opens on conflict', 'Opens on setup, not conflict',
          'Conflict or danger shows in the first two lines, not a greeting or setup.',
          "First-time viewer (AI) for shortlisted clips; the opening line's type before that"),
    Check('context_free_open', 'Opening line needs no backstory', "Opening line names someone a stranger can't place",
          "The first line names no person or past event a stranger couldn't place.",
          "Code, from the AI's notes on that line"),
    Check('small_cast', '3 or fewer speakers', '4 or more speakers',
          'At most 3 people speak.',
          'Code'),
    Check('ends_open', 'Stops before the payoff', 'Answers its own question before the cut',
          'Nothing releases the tension after the last escalation, and the first-time viewer (AI) says the question is still open.',
          'Code and the first-time viewer (AI)'),
    Check('builds', 'Builds to the end', 'Loses intensity at the end',
          'The last third is at least as intense as the middle third (AI intensity scores).',
          'Code'),
    Check('no_dead_air', 'No silence over 4 s', 'A silence over 4 s',
          'No gap over 4 s between lines or actions.',
          'Code'),
    Check('features_lead', 'Features a lead', 'No lead on screen',
          'One of the two leads ({leads}) speaks or appears.',
          "Code (leads from the AI's reading of the show)"),
    Check('stakes_clear', 'Stakes clear to a stranger', 'Stakes unclear to a stranger',
          'The first-time viewer (AI) can fill in "X wants Y, but Z blocks it". It passes whenever the AI writes an answer.',
          'First-time viewer (AI)'),
    Check('rootable', 'Someone to root for', 'Nobody to root for',
          'The first-time viewer (AI) names someone to side with. It passes whenever the AI writes a name.',
          'First-time viewer (AI)'),
)}


def term(name: str) -> str:
    """GLOSSARY[name]. Raises KeyError on an unknown term, so tests catch drift."""
    return GLOSSARY[name]


def penalty(cfg: dict, key: str) -> float:
    """cfg["rubric"]["penalties"][key]; 1.0 if absent."""
    try:
        return float(((cfg or {}).get("rubric") or {}).get("penalties", {})[key])
    except (KeyError, TypeError, ValueError):
        return 1.0


TYPE_WORDS = {"slap": "a slap", "threat": "a threat", "reveal": "a reveal", "humiliation": "a humiliation",
              "romance": "a romantic line", "line": "a loaded line", "exposition": "exposition",
              "other": "the same kind of moment"}
HOOK_TYPE_SHOWN = {"slap", "threat", "reveal", "humiliation", "romance"}  # "line", "other", "exposition" are omitted
ROLE_WORDS = {"setup": "sets up", "escalation": "raises the pressure", "turn": "turns the power",
              "payoff": "releases the tension"}
PART_NAMES = {"opening": "Opening", "middle": "Middle", "late": "Late"}
TP_WORDS = {"opportunity": "opportunity", "change_of_plans": "change of plans", "point_of_no_return": "point of no return",
            "major_setback": "major setback", "climax": "climax"}
# key -> (button, past tense, badge colour, icon, shortcut)
DECISIONS = {"approve": ("Approve", "Approved", "green", ":material/check_circle:", "A"),
             "trim": ("Trim", "Trim", "orange", ":material/content_cut:", "T"),
             "reject": ("Reject", "Rejected", "gray", ":material/block:", "X")}
REJECT_REASONS = ["Confusing out of context", "Weak hook", "Resolves on screen", "Gives away a twist", "Bad cut point",
                  "Repeats another promo", "Other"]
