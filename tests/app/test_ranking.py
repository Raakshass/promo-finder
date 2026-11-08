"""Lane R: replay, sentences, rule box, map arrays and the Ranking view (checks 23-27, 29).

Use the fixtures official_dir and app_test (conftest.py), helpers from apphelpers.py and literal values from
expected_official.py. Derive numbers from the run files where you can."""
from __future__ import annotations

import copy
import re
from pathlib import Path

import expected_official as X
import pytest
from apphelpers import find_evidence, rendered_text

from app.ui import checks, explain_rank, fmt, showmap
from app.ui import data as D

APP_DIR = Path(__file__).resolve().parents[2] / "app"


@pytest.fixture(scope="module")
def run(official_dir):
    return D.load_run(str(official_dir), D.stamp(official_dir))


@pytest.fixture(scope="module")
def parts_(run):
    return explain_rank.parts(run["promos"], run["cfg"], run["episodes_list"])


def _by_rank(run):
    return {p["rank"]: p for p in run["promos"]}


# ---------------------------------------------------------------- parts (used by Promos card 2)

def test_parts_match_slate(run, parts_):
    got = tuple((p.key, p.first_ep, p.last_ep, p.n) for p in parts_)
    assert got == X.PARTS
    assert [p.name for p in parts_] == ["Opening", "Middle", "Late"]
    assert parts_[0].first_rank == 1 and parts_[-1].last_rank == len(run["promos"])
    blocks = run["metrics"]["editor_rubric"]["slate_blocks"]
    assert {p.key: p.n for p in parts_} == blocks


# ---------------------------------------------------------------- check 23: the replay

def test_replay_reproduces_ranks(run):
    cfg = run["cfg"]
    assert explain_rank.replayable(run["promos"], cfg)
    pl = explain_rank.replay(run["promos"], cfg)
    assert len(pl) == len(run["promos"])
    sc = cfg["scoring"]
    for part in ("opening", "middle", "late"):
        mine = sorted((p for p in run["promos"] if p["flags"]["block"] == part), key=lambda p: p["rank"])
        placed = []
        remaining = list(mine)
        while remaining:
            def counted(q):
                same = sum(1 for x in placed if x["flags"]["episode"] == q["flags"]["episode"])
                if len(placed) < 10 and same >= sc["max_per_episode_top10"]:
                    return None
                v = q["score"] * (1 - sc["same_episode_penalty"]) ** same
                if placed and placed[-1]["flags"].get("hook_type") == q["flags"].get("hook_type"):
                    v *= 1 - sc["same_type_penalty"]
                return v
            vals = [(counted(q), q) for q in remaining]
            best = max((v for v, _ in vals if v is not None))
            pick = min((q for v, q in vals if v is not None and v == best), key=lambda q: q["rank"])
            assert pick is remaining[0], f"replay would place #{pick['rank']} before #{remaining[0]['rank']}"
            assert pl[pick["promo_id"]].counted == pytest.approx(best)
            placed.append(pick)
            remaining.remove(pick)


def test_replay_first_in_part(run):
    pl = explain_rank.replay(run["promos"], run["cfg"])
    firsts = sorted(p.rank for p in pl.values() if p.first_in_part)
    assert firsts == [1, 1 + X.PARTS[0][3], 1 + X.PARTS[0][3] + X.PARTS[1][3]]


def test_not_replayable_without_block(run):
    promos = copy.deepcopy(run["promos"])
    promos[0]["flags"].pop("block")
    assert not explain_rank.replayable(promos, run["cfg"])
    assert explain_rank.replay(promos, run["cfg"]) == {}


# ---------------------------------------------------------------- check 24: sentences

def test_why_sentences(run, parts_):
    pl = explain_rank.replay(run["promos"], run["cfg"])
    r = _by_rank(run)
    why = {k: explain_rank.why_sentence(p["promo_id"], pl, run["promos"], parts_) for k, p in r.items()}
    assert X.WHY_19 in why[19]
    assert X.WHY_4 in why[4]
    assert X.WHY_12 in why[12]
    assert why[1].startswith("Posted first: the highest score in episodes 1–10")
    assert X.WHY_3_FULL in why[3]
    for k, (score, counted) in X.WHY_3_RIVALS.items():
        assert f"#{k} ({score})" in why[3] and counted in why[3]
    for s in why.values():
        assert s and "None" not in s and not re.search(r"e\d{3}-b\d{2}", s) and "P0" not in s


# ---------------------------------------------------------------- check 25: the rule box reads the run

def test_rule_lines_official(run, parts_):
    text = "\n".join(explain_rank.rule_lines(run, parts_))
    assert "episodes 1–10 take the first 11 places, episodes 11–25 the next 7 and episodes 26–50 the last 2" in text
    assert "55 / 33 / 12%" in text and "501 ads from 28 dramas" in text
    assert "×0.85" in text and "×0.90" in text and "At most 2 promos per episode" in text
    assert "#19 has the highest score of all 20 (0.945) but comes from episode 31" in text


def test_rule_lines_follow_config(run, parts_):
    d = copy.deepcopy(run)
    d["cfg"]["slate"]["shares"] = {"opening": 0.6, "middle": 0.3, "late": 0.1}
    d["cfg"]["scoring"]["same_episode_penalty"] = 0.2
    d["cfg"]["scoring"]["max_per_episode_top10"] = 3
    text = "\n".join(explain_rank.rule_lines(d, parts_))
    assert "60 / 30 / 10%" in text and "×0.80" in text and "At most 3 promos" in text
    assert "55 / 33 / 12%" not in text and "×0.85" not in text


def test_rule_lines_without_dramaad(run, parts_):
    d = dict(run, dramaad={})
    text = "\n".join(explain_rank.rule_lines(d, parts_))
    assert "DramaAD dataset; see Method" in text


def test_no_share_or_penalty_literals_in_lane_files(run):
    """Check 25 for Lane R's files: shares and penalties are read, never typed."""
    cfg = run["cfg"]
    values = {cfg["slate"]["shares"][b] for b in ("opening", "middle", "late")}
    values |= {cfg["scoring"]["same_episode_penalty"], cfg["scoring"]["same_type_penalty"]}
    pats = {re.escape(repr(float(v))) for v in values}
    pats |= {re.escape(fmt.mult(v)) for v in cfg["rubric"]["penalties"].values()}
    pats |= {re.escape(fmt.mult(1 - cfg["scoring"]["same_episode_penalty"])),
             re.escape(fmt.mult(1 - cfg["scoring"]["same_type_penalty"]))}
    for f in ("ui/explain_rank.py", "ui/showmap.py", "ui/checks.py", "views/ranking.py", "views/quality.py"):
        src = (APP_DIR / f).read_text()
        for pat in pats:
            assert not re.search(rf"(?<![\d.])(?<!p = ){pat}(?!\d)", src), f"{f} types {pat}"  # p-values aside


# ---------------------------------------------------------------- check 27: map arrays

def test_map_arrays_shape_and_hover(official_dir, run):
    a = showmap.map_arrays(str(official_dir), D.stamp(official_dir))
    E = len(run["episodes_list"])
    assert len(a["eps"]) == E and len(a["z_hook"]) == 2 * E and len(a["z_cliff"]) == 2 * E
    assert a["max_s"] == int(-(-max(a["durations"]) // 1))
    beat_re = re.compile(r"e\d{3}-b\d{2}")
    line_re = re.compile(r"^Ep \d+ · \d+:\d{2}\.\d–\d+:\d{2}\.\d · .+ · hook [\d.]+/3 · cliffhanger [\d.]+/3")
    seen_line = 0
    for i in range(E):
        dur = a["durations"][i]
        for row in (2 * i, 2 * i + 1):
            for t, h in enumerate(a["hover_hook"][row] if row % 2 == 0 else a["hover_cliff"][row]):
                assert not beat_re.search(h) and "None" not in h
                if t >= int(-(-dur // 1)):
                    continue
                if "no line" not in h:
                    assert line_re.match(h), h
                    seen_line += 1
        # lanes: hook values only on even rows, cliffhanger values only on odd rows
        assert all(v is None for v in a["z_cliff"][2 * i]) and all(v is None for v in a["z_hook"][2 * i + 1])
        assert a["z_hook"][2 * i][-1] is None or int(-(-dur // 1)) >= len(a["z_hook"][2 * i])
    assert seen_line > 1000
    # the promo appears in the hover of its own cells
    p5 = _by_rank(run)[5]
    s = p5["segments"][0]
    i = a["eps"].index(s["episode"])
    t = int(s["start_s"]) + 2
    assert "in promo #5" in a["hover_hook"][2 * i][t]


# ---------------------------------------------------------------- check 26 (data side): figure

def test_figure_spec(run, parts_):
    fig = showmap.figure(run, parts_)
    heat = [t for t in fig.data if t.type == "heatmap"]
    assert len(heat) == 2 and all(t.zmin == -0.5 and t.zmax == 3.5 and not t.showscale for t in heat)
    badges = [t for t in fig.data if t.type == "scatter"][0]
    assert len(badges.x) == X.SEGMENTS
    filled = [txt for txt, c in zip(badges.text, badges.marker.color) if c == "#111118"]
    assert sorted({int(x) for x in filled}) == list(range(1, 11))
    notes = [a.text for a in fig.layout.annotations]
    for lab in X.SEPARATOR_LABELS:
        assert lab in notes
    flat_y = {a.y for a in fig.layout.annotations if a.text == "AI scored every line 0, twice"}
    eps = run["episodes_list"]
    assert flat_y == {2 * eps.index(e) + 0.5 for e in X.FLAT_NOTE_EPS}
    seps = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(seps) == len(parts_) - 1
    # no text under 12 px
    sizes = [fig.layout.font.size, fig.layout.xaxis.tickfont.size, fig.layout.yaxis.tickfont.size,
             badges.textfont.size] + [a.font.size for a in fig.layout.annotations]
    assert min(sizes) >= 12
    assert fig.layout.height == 18 * len(eps) + 120
    hovers = list(badges.hovertext)
    assert all("None" not in h and "P0" not in h for h in hovers)


def test_takeaways(run, parts_):
    tk = showmap.takeaways(run, parts_)
    assert any(t.startswith(X.TAKEAWAY_1) for t in tk)
    assert any("Rows 8 and 17 stay grey" in t and "Row 35 has no line" in t for t in tk)
    assert any(" of 50 rows the strongest cliffhanger sits in the last 10 seconds" in t for t in tk)


def test_colour_scales_are_stepped():
    assert showmap._scale(("a", "b", "c", "d")) == [[0, "a"], [.25, "a"], [.25, "b"], [.5, "b"], [.5, "c"],
                                                    [.75, "c"], [.75, "d"], [1, "d"]]


def test_legend_has_words_in_ink():
    html = showmap.legend_html()
    assert "Top lane" in html and "Bottom lane" in html and "font-size:13px" in html
    assert not re.search(r"font-size:(\d|1[01])px", html)


# ---------------------------------------------------------------- checks used by Promos and the drawer

def test_before_posting_official(run):
    flagged = sorted(p["rank"] for p in run["promos"] if checks.before_posting(p, run))
    assert tuple(flagged) == X.TO_CHECK_RANKS
    r = _by_rank(run)
    assert [f.short for f in checks.before_posting(r[3], run)] == ["Scene change"]
    assert [f.short for f in checks.before_posting(r[9], run)] == ["Join", "Answer not verified"]
    assert checks.before_posting(r[9], run)[1].text.startswith(X.PROMISE_9)
    assert [f.short for f in checks.before_posting(r[10], run)] == ["Not a cliffhanger"]
    assert checks.fail_count(run) == X.FAIL_COUNT


def test_weak_spots_and_badge(run):
    r = _by_rank(run)
    ws = checks.weak_spots(r[3], run)
    assert ws[0].startswith("Fails 2 of 9 editor's checks:") and ws[0].endswith("(score ×0.81)")
    assert any(w.startswith("Fragile pick: on the 40-clip shortlist in only 42% of 60") for w in checks.weak_spots(r[9], run))
    assert checks.stability_badge(r[2], run)[:2] == ("Steady pick · 100%", "gray")
    assert checks.stability_badge(r[9], run)[:2] == ("Fragile pick · 42%", "orange")
    assert checks.stability_badge(r[2], dict(run, stability={})) is None


# ---------------------------------------------------------------- check 29: the Ranking view (AppTest)

def test_ranking_view(app_test):
    at = app_test
    at.session_state["view"] = "Ranking"
    at.run()
    assert not at.exception
    text = rendered_text(at)
    assert X.TAKEAWAY_1 in text
    heads = [m.value for m in at.markdown if m.value.startswith("##### ")]
    assert len(heads) == 3 and heads[0] == "##### Opening · episodes 1–10 · 11 places"
    opens = [b for b in at.button if b.key and b.key.startswith("rk-")]
    assert len(opens) == X.N_PROMOS
    assert "How the order is set (plain code, no AI)" in text
    for bad in ("None", "nan", "0/0", "{'"):
        assert bad not in text.replace("Nonetheless", "")
    assert not re.search(r"e\d{3}-b\d{2}", text)


def test_ranking_open_button_opens_drawer(app_test):
    at = app_test
    at.session_state["view"] = "Ranking"
    at.run()
    at.button(key="rk-4").click().run()
    assert at.session_state["open_pid"] == "P04"


def test_ranking_keyword_run(app_test):
    if find_evidence("heuristic_only") is None:
        pytest.skip("no keyword-rules run")
    at = app_test
    at.session_state["view"] = "Ranking"
    at.query_params["run"] = "keyword-rules"
    at.run()
    assert not at.exception
    assert "Scores here come from keyword rules, not the AI." in rendered_text(at)


def test_ranking_first_version(app_test):
    if find_evidence("previous_version") is None:
        pytest.skip("no first-version run")
    at = app_test
    at.session_state["view"] = "Ranking"
    at.query_params["run"] = "first-version"
    at.run()
    assert not at.exception
    text = rendered_text(at)
    assert "None" not in text
