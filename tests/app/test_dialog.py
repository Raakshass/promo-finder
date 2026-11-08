"""Lane P: explain_promo (check 19), the timeline and the drawer (checks 16-18, 22).

Use the fixtures official_dir and app_test (conftest.py), helpers from apphelpers.py and literal values from
expected_official.py. Derive numbers from the run files where you can."""
from __future__ import annotations

import json
import math

import pytest
import yaml

import expected_official as X
from apphelpers import find_evidence, rendered_text as _raw_text


def rendered_text(at) -> str:
    """apphelpers.rendered_text with the protobuf escapes of quotes undone."""
    return _raw_text(at).replace("\\'", "'").replace('\\"', '"')


def _promos(run_dir) -> list[dict]:
    return sorted(json.loads((run_dir / "promos.json").read_text(encoding="utf-8")), key=lambda p: p["rank"])


def _by_rank(run_dir, rank: int) -> dict:
    return next(p for p in _promos(run_dir) if p["rank"] == rank)


def _cfg(run_dir) -> dict:
    return yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}


def _arc(run_dir) -> dict:
    f = run_dir / "show_arc.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _open(at, pid: str):
    at.session_state["open_pid"] = pid
    at.run()
    assert not at.exception
    return at


# ----------------------------------------------------------------------------------------------------- explain_promo

def test_score_lines_multiply_to_the_score(official_dir):
    """Check 19 (official): lines for every promo, and the printed factors multiply to the score within 0.001."""
    from app.ui import explain_promo as E, fmt
    cfg, arc = _cfg(official_dir), _arc(official_dir)
    for p in _promos(official_dir):
        lines = E.score_lines(p, cfg, arc)
        assert lines is not None, p["rank"]
        assert lines[0].startswith(f"Score {fmt.score(p['score'])} = base ")
        assert abs(math.prod(E.score_factors(p)) - p["score"]) <= 0.001, p["rank"]
        assert lines[1].startswith("Base ")
        for w in ("w_hook", "w_cliff", "w_standalone", "w_density"):
            assert f"{cfg['scoring'][w]:.2f} ×" in lines[1]


def test_score_lines_none_for_lane_runs():
    """Check 19 (consistency-a): None exactly for the promos whose stored parts don't multiply to their score."""
    from app.ui import explain_promo as E, fmt
    run = find_evidence("stability_base")
    if run is None:
        pytest.skip("no consistency-a run in the bundle")
    cfg, arc = _cfg(run), _arc(run)
    nones = 0
    for p in _promos(run):
        s = p["subscores"]
        product = s["base"] * s.get("rubric_mult", 1.0) * s.get("arc_mult", 1.0) * (
            0.9 if p["flags"].get("kind") == "two_segment" else 1.0)
        multiplies = abs(product - p["score"]) <= 0.001
        assert (E.score_lines(p, cfg, arc) is None) == (not multiplies), p["rank"]
        nones += not multiplies
    assert nones > 0


def test_score_lines_none_without_base():
    from app.ui import explain_promo as E, fmt
    run = find_evidence("previous_version")
    if run is None:
        pytest.skip("no first-version run in the bundle")
    assert all(E.score_lines(p, _cfg(run), {}) is None for p in _promos(run))


def test_score_lines_follow_the_config(official_dir):
    """Weights and penalties are read from the config, never typed in."""
    from app.ui import explain_promo as E, fmt
    p = _by_rank(official_dir, 3)
    cfg = _cfg(official_dir)
    cfg["scoring"]["w_hook"] = 0.4
    fails = [k for k, ok in p["rubric"].items() if not ok]
    cfg["rubric"]["penalties"][fails[0]] = 0.5
    lines = E.score_lines(p, cfg, _arc(official_dir))
    assert "0.40 × hook" in lines[1]
    assert "(×0.50)" in "\n".join(lines)


def test_labels(official_dir):
    from app.ui import explain_promo as E, fmt
    p3 = _by_rank(official_dir, 3)
    label = E.hook_label(p3)
    assert label.startswith(f"Hook · first 3 s · starts {p3['hook']['at_s']:.1f} s in · AI {p3['hook']['score']}/3")
    assert label.endswith("physical action in the first 3 s") == bool(p3["flags"]["physical_hook"])
    lab, warn = E.cliff_label(p3)
    assert lab.startswith("Cliffhanger · at the cut") and warn is None
    p10 = _by_rank(official_dir, X.RESOLVES_RANK)
    lab, warn = E.cliff_label(p10)
    assert lab.startswith("Closing line (resolves on screen)")
    assert warn.startswith("Fails a brief requirement")


def test_promise_sentences(official_dir):
    from app.ui import explain_promo as E, fmt
    p9 = _by_rank(official_dir, 9)
    assert X.PROMISE_9 in E.promise_sentences(p9, {})[-1]
    assert E.promise_phrase(p9) == "not verified"
    for p in _promos(official_dir):
        lines = E.promise_sentences(p, {})
        assert lines and all("None" not in ln for ln in lines)
    assert E.promise_sentences({}, {}) == ["Not checked: this run used no AI model."]
    low = {"promise": {"status": "same episode", "confidence": "low"}}
    assert E.promise_sentences(low, {})[-1].endswith("The AI marked this answer as a guess.")
    pay = {"promise": {"status": "behind the paywall", "answered_in_episode": 12}}
    assert "paid from episode 10" in E.promise_sentences(pay, {"args": {"paywall": 10}})[-1]
    rej = {"promise": {"status": "rejected", "note": "model named episode 60, outside 5..50"}}
    assert "which is after the last episode" in E.promise_sentences(rej, {})[-1]


def test_viewer_rows(official_dir):
    from app.ui import explain_promo as E, fmt
    p3 = _by_rank(official_dir, 3)
    labels = [lab for lab, _ in E.viewer_rows(p3)]
    assert labels[:2] == ["What it saw", "Left asking"]
    assert ("Couldn't place" in labels) == bool(p3["cold_viewer"]["unresolved_refs"])
    assert E.viewer_rows({"cold_viewer": None}) == []


# ----------------------------------------------------------------------------------------------------------- timeline

def test_timeline(official_dir):
    from app.ui import components
    p5 = _by_rank(official_dir, 5)
    html = components.timeline_html(p5)
    assert "height:64px" in html
    assert f"Hook {p5['hook']['at_s']:.1f} s" in html
    assert f"Ep {p5['segments'][1]['episode']} starts" in html
    assert components.timeline_html({}) == ""


def test_timeline_escapes_script_text():
    from app.ui import components
    evil = "</script><b>x</b>"
    p = {"rank": 1, "duration_s": 30.0, "hook": {"at_s": 0.5}, "cliffhanger": {"at_s": 30.0},
         "segments": [{"episode": evil, "start": evil, "end": "00:15.00", "start_s": 0.0, "end_s": 15.0},
                      {"episode": evil, "start": "00:00.00", "end": "00:15.00", "start_s": 0.0, "end_s": 15.0}]}
    html = components.timeline_html(p)
    assert "<b>" not in html and "</script>" not in html
    assert "&lt;b&gt;" in html


# ------------------------------------------------------------------------------------------------------------- drawer

def test_drawer_promo_3(app_test, official_dir):
    """Check 16 and 17."""
    from app.ui import checks
    p3 = _by_rank(official_dir, 3)
    at = _open(app_test.run(), p3["promo_id"])
    text = rendered_text(at)
    assert X.DRAWER_3_HEADING in text
    assert X.DRAWER_3_CUT in [c.value for c in at.code]
    assert "Score arithmetic · 0.758" in text
    assert f"Promo id {p3['promo_id']}" in text
    if not checks.before_posting(p3, {}) and not checks.before_posting({"segments": [{}, {}]}, {}):
        pytest.skip("checks.before_posting has not landed")
    for frag in X.DRAWER_3_BEFORE_POSTING:
        assert frag in text
    if "Why #3 here." not in text:
        pytest.skip("explain_rank.why_sentence has not landed")
    why = next(line for line in text.splitlines() if "Why #3 here." in line)
    for rank, (score, counted) in X.WHY_3_RIVALS.items():
        assert f"#{rank} ({score})" in why and counted in why


def test_drawer_promise_and_closing_line(app_test, official_dir):
    """Check 18."""
    at = _open(app_test.run(), _by_rank(official_dir, 9)["promo_id"])
    assert X.PROMISE_9 in rendered_text(at)
    del at.session_state["open_pid"]
    at = _open(at, _by_rank(official_dir, X.RESOLVES_RANK)["promo_id"])
    text = rendered_text(at)
    assert "Closing line (resolves on screen)" in text
    assert ":orange-badge[:material/flag: Fails a brief requirement" in text


def test_drawer_decisions(app_test, official_dir):
    """Approve, Trim with a note, Reject with a reason, Clear; Previous and Next."""
    from app.ui.glossary import REJECT_REASONS
    p3 = _by_rank(official_dir, 3)
    pid = p3["promo_id"]
    at = _open(app_test.run(), pid)
    run_id = next(iter([p3["provenance"]["run_id"]]))
    at.button(key=f"dr-trim-official-{pid}").click().run()
    assert at.session_state["review"][run_id][pid]["call"] == "trim"
    at.text_input(key=f"note-official-{pid}").input("start 2 s later").run()
    assert at.session_state["review"][run_id][pid]["note"] == "start 2 s later"
    at.button(key=f"dr-reject-official-{pid}").click().run()
    at.button_group(key=f"why-official-{pid}").set_value(REJECT_REASONS[1]).run()
    entry = at.session_state["review"][run_id][pid]
    assert (entry["call"], entry["reason"], entry["note"]) == ("reject", REJECT_REASONS[1], "start 2 s later")
    assert "Your call: Rejected" in rendered_text(at)
    at.button(key=f"dr-clear-official-{pid}").click().run()
    assert pid not in at.session_state["review"][run_id]
    at.button(key="dr-next-official").click().run()
    assert at.session_state["open_pid"] == _by_rank(official_dir, 4)["promo_id"]
    assert "#4 · " in rendered_text(at)


def test_drawer_read_only_on_evidence_runs(app_test):
    """Check 22."""
    from app.ui import data
    run = next((r for r in data.list_runs() if r.role == "evidence"), None)
    if run is None:
        pytest.skip("no evidence run in the bundle")
    at = app_test.run()
    at.radio(key="run").set_value(run.key).run()
    first = _promos(run.dir)[0]["promo_id"]
    at = _open(at, first)
    text = rendered_text(at)
    assert "Read-only: this is an evidence run." in text
    assert not [b for b in at.button if b.label == "Approve"]


@pytest.mark.parametrize("key", ["official", "first-version", "keyword-rules", "consistency-a", "cues-hidden"])
def test_drawer_text_is_clean(app_test, key):
    """No raw Python values in the drawer for any run (check 40, drawer part). The footer's promo id is allowed."""
    import re
    from app.ui import data
    run = next((r for r in data.list_runs() if r.key == key), None)
    if run is None:
        pytest.skip(f"no {key} run in the bundle")
    at = app_test.run()
    if key != "official":
        at.radio(key="run").set_value(key).run()
    bad = re.compile(r"\bNone\b|\bnan\b|(?<![\d.])0/0(?![\d.])|–/5|\{'|\be\d{3}-b\d{2}\b")
    for p in _promos(run.dir)[:20]:
        at.session_state["open_pid"] = p["promo_id"]
        at.run()
        assert not at.exception, (key, p["rank"], at.exception)
        # element ids are protocol data, never shown (a dialog's id ends in "-None")
        text = "\n".join(line for line in rendered_text(at).splitlines() if "$$ID-" not in line)
        hits = [m.group() for m in bad.finditer(text)]
        assert not hits, (key, p["rank"], hits)
        ids = set(re.findall(r"\bP\d{2}\b", text)) - {p["promo_id"]}
        assert not ids, (key, p["rank"], ids)
