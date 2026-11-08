"""Lane P: exports (checks 12-13) and the Promos view (checks 4, 8-11, 14-15).

Use the fixtures official_dir and app_test (conftest.py), helpers from apphelpers.py and literal values from
expected_official.py. Derive numbers from the run files where you can."""
from __future__ import annotations

import csv
import io
import json

import pytest

import expected_official as X
from apphelpers import find_evidence, rendered_text as _raw_text


# ------------------------------------------------------------------------------------------------------------ helpers

def _data(run_dir):
    from app.ui import data
    return data.load_run(str(run_dir), data.stamp(run_dir))


def _promos(run_dir) -> list[dict]:
    return sorted(json.loads((run_dir / "promos.json").read_text(encoding="utf-8")), key=lambda p: p["rank"])


def _pid(run_dir, rank: int) -> str:
    return next(p["promo_id"] for p in _promos(run_dir) if p["rank"] == rank)


def rendered_text(at) -> str:
    """apphelpers.rendered_text with the protobuf escapes of quotes undone, so copy can be matched as written."""
    return _raw_text(at).replace("\\'", "'").replace('\\"', '"')


def _official_run():
    from app.ui import data
    return next(r for r in data.list_runs() if r.key == "official")


def _table(at):
    return next(df for df in (d.value for d in at.dataframe) if "#" in df.columns)


# ------------------------------------------------------------------------------------------------------------ exports

def test_cut_list_is_the_runs_promos_csv_plus_three_columns(official_dir):
    """Check 12: one row per segment; the run's own columns byte-identical; your_call, trim_note, reject_reason."""
    from app.ui import exports
    out = exports.cut_list_csv(_data(official_dir), {})
    raw = (official_dir / "promos.csv").read_bytes().decode("utf-8")
    src_lines = raw.split("\r\n")
    out_lines = out.split("\r\n")
    src_rows = list(csv.reader(io.StringIO(raw, newline="")))
    rows = list(csv.reader(io.StringIO(out, newline="")))
    assert len(rows) - 1 == sum(len(p["segments"]) for p in _promos(official_dir)) == X.SEGMENTS
    assert rows[0][len(src_rows[0]):] == ["your_call", "trim_note", "reject_reason"]
    assert len(src_rows[0]) == 17 and rows[0][:17] == src_rows[0]
    for src, line in zip(src_lines, out_lines):
        if src:
            assert line.startswith(src + ","), "the run's own columns must stay byte-identical"
    pid5 = _pid(official_dir, 5)
    five = [(r[3], r[4], r[5]) for r in rows[1:] if r[1] == pid5]
    want = [(str(s["episode"]), s["start"], s["end"]) for s in _promos(official_dir)[4]["segments"]]
    assert five == want and len(five) == 2
    assert all(r[-3:] == ["", "", ""] for r in rows[1:])


def test_cut_list_only_kept(official_dir):
    """Check 13: #1 approved, #3 trim, only_kept -> exactly the rows of #1 and #3, with the decision columns."""
    from app.ui import exports
    p1, p3, p4 = _pid(official_dir, 1), _pid(official_dir, 3), _pid(official_dir, 4)
    review = {p1: {"call": "approve", "note": "", "reason": ""},
              p3: {"call": "trim", "note": "start 2 s later", "reason": ""},
              p4: {"call": "reject", "note": "", "reason": "Weak hook"}}
    rows = list(csv.reader(io.StringIO(exports.cut_list_csv(_data(official_dir), review, only_kept=True), newline="")))
    body = rows[1:]
    n_segs = {p["promo_id"]: len(p["segments"]) for p in _promos(official_dir)}
    assert sorted({r[1] for r in body}) == sorted([p1, p3])
    assert len(body) == n_segs[p1] + n_segs[p3]
    assert {r[1]: tuple(r[-3:]) for r in body}[p3] == ("trim", "start 2 s later", "")
    every = list(csv.reader(io.StringIO(exports.cut_list_csv(_data(official_dir), review), newline="")))[1:]
    assert {r[1]: tuple(r[-3:]) for r in every}[p4] == ("reject", "", "Weak hook")


def test_cut_list_rebuilt_from_json_when_csv_missing(official_dir):
    from app.ui import exports
    d = dict(_data(official_dir), promos_csv="")
    rows = list(csv.reader(io.StringIO(exports.cut_list_csv(d, {}), newline="")))
    src = list(csv.reader(io.StringIO((official_dir / "promos.csv").read_bytes().decode("utf-8"), newline="")))
    assert rows[0][:17] == src[0]
    assert len(rows) == len(src)
    assert [r[:7] for r in rows[1:]] == [r[:7] for r in src[1:]]   # rank, id, segment, episode, in, out, length


def test_review_csv_only_decided(official_dir):
    from app.ui import exports
    run = _official_run()
    p3 = _pid(official_dir, 3)
    out = exports.review_csv(run, _data(official_dir), {p3: {"call": "reject", "note": "n", "reason": "Weak hook",
                                                             "at": "2026-11-08T10:00:00+05:30"}})
    rows = list(csv.reader(io.StringIO(out, newline="")))
    assert rows[0] == ["run_id", "show", "promo_id", "rank", "segments", "your_call", "trim_note", "reject_reason",
                       "decided_at"]
    assert len(rows) == 2
    assert rows[1][:4] == [run.run_id, run.show_title, p3, "3"]
    assert rows[1][5:] == ["reject", "n", "Weak hook", "2026-11-08T10:00:00+05:30"]


def test_timecodes_text(official_dir):
    from app.ui import exports
    text = exports.timecodes_text(_promos(official_dir))
    lines = text.splitlines()
    assert len(lines) == X.SEGMENTS
    i = next(k for k, ln in enumerate(lines) if ln.startswith("#5 "))
    s1, s2 = _promos(official_dir)[4]["segments"]
    assert lines[i].startswith(f"#5  Ep {s1['episode']}  {s1['start']}–{s1['end']}  ")
    assert lines[i + 1].startswith(f"    Ep {s2['episode']}  {s2['start']}–{s2['end']}  ")
    assert "\t" not in text


# ------------------------------------------------------------------------------------------------------- Promos view

def test_cards_and_table(app_test, official_dir):
    """Checks 2 (cards), 8, 9 and 10."""
    from app.ui import fmt
    at = app_test.run()
    assert not at.exception
    text = rendered_text(at)
    assert X.CARD_HEADLINES[0] in text
    assert not at.slider
    df = _table(at)
    promos = _promos(official_dir)
    assert df["#"].tolist() == [f"Open #{k}" for k in range(1, len(promos) + 1)]
    assert len(df) == X.N_PROMOS
    assert list(df["Segments (episode time)"].iloc[4]) == X.SEGMENTS_OF_5
    for row, p in zip(df.to_dict("records"), promos):
        assert list(row["Segments (episode time)"]) == fmt.segs(p)
        for col in ("Hook (first 3 s)", "Cliffhanger (at the cut)", "Why it works"):
            assert row[col].strip(), (p["rank"], col)
    assert df["Cliffhanger (at the cut)"].iloc[X.RESOLVES_RANK - 1].startswith("Resolves: ")
    assert sum(c.startswith("Resolves: ") for c in df["Cliffhanger (at the cut)"]) == sum(
        1 for p in promos if p["rubric"].get("ends_open") is False)
    assert df["Checks"].iloc[2] == X.CHECKS_CELL[3] and df["Checks"].iloc[9] == X.CHECKS_CELL[10]
    assert "Your review: 0 of 20 decided" in text
    assert "Ranked lower, not removed." in text and "random clips" not in text   # no benchmark claim on the card
    assert f"All {X.N_PROMOS} promos run" in text
    assert "Post from the top" in text


def test_card_headlines_use_lane_r(app_test, official_dir):
    """Cards 2 and 3 read explain_rank.parts and checks.fail_count (Lane R)."""
    from app.ui import checks, explain_rank
    d = _data(official_dir)
    if not explain_rank.parts(d["promos"], d["cfg"], d["episodes_list"]) or not checks.fail_count(d):
        pytest.skip("explain_rank.parts or checks.fail_count has not landed")
    at = app_test.run()
    text = rendered_text(at)
    for headline in X.CARD_HEADLINES:
        assert headline in text
    assert "Post from the top: #1–#11 come from episodes 1–10, #12–#18 from 11–25, #19–#20 from 26–50." in text


@pytest.mark.parametrize("key, view", [("card-find", "Ranking"), ("card-rank", "Ranking"), ("card-quality", "Quality")])
def test_card_buttons_switch_view(app_test, key, view):
    """Check 4."""
    at = app_test.run()
    at.button(key=key).click().run()
    assert at.session_state["view"] == view


def test_to_check_filter(app_test):
    """Check 11."""
    from app.ui import checks
    at = app_test.run()
    at.button_group(key="promos_filter").set_value("check").run()
    assert not at.exception
    text = rendered_text(at)
    if "No promo matches this filter." in text and not checks.before_posting({"segments": [{}, {}]}, {}):
        pytest.skip("checks.before_posting has not landed")
    df = _table(at)
    assert df["#"].tolist() == [f"Open #{k}" for k in X.TO_CHECK_RANKS]
    assert all(len(v) for v in df["To check"])


def test_decision_filters_and_open_from_table(app_test, official_dir, monkeypatch):
    import streamlit as st
    from app.ui import state
    from app.views import promos as view
    at = app_test.run()
    at.button_group(key="promos_filter").set_value("approve").run()
    assert "No promo matches this filter." in rendered_text(at)
    # the table's Open button callback parses the rank from the clicked label (robust to sorting)
    fake = {"promos_open": {"row": 0, "label": "Open #5"}}
    monkeypatch.setattr(st, "session_state", fake)
    view._open_from_table({5: _pid(official_dir, 5)})
    assert fake.get(state.OPEN) == _pid(official_dir, 5)


def test_best_clip_list(app_test, official_dir):
    """Check 14 and the best-clip caption."""
    from app.ui import checks
    at = app_test.run()
    at.button_group(key="promos_list").set_value("episodes").run()
    assert not at.exception
    df = next(d.value for d in at.dataframe if "Episode" in d.value.columns)
    rows = {int(r["Episode"]): r for r in df.to_dict("records")}
    with_clip = sum(1 for r in csv.DictReader(io.StringIO((official_dir / "promos_by_episode.csv").read_text())) if r["in"])
    assert with_clip == X.BEST_CLIP_WITH_CLIP
    assert f"{with_clip} of {X.EPISODES} episodes have one; {X.BEST_CLIP_PASS_ALL} of those pass" in rendered_text(at)
    empty = X.NO_CLIP_ALL_ZERO + X.NO_CLIP_FLAT + X.NO_CLIP_NO_FIT
    for ep in empty:
        assert rows[ep]["Hook"].startswith("No clip"), ep
        assert rows[ep]["Length"] == "" and rows[ep]["In–out"] == ""
    assert len(rows) == X.EPISODES and all(r["Part"] for r in rows.values())
    import re
    text = "\n".join(line for line in rendered_text(at).splitlines() if "$$ID-" not in line)
    assert not re.findall(r"\bNone\b|\bnan\b|\{'|\be\d{3}-b\d{2}\b", text)
    if not checks.flat_episodes(_data(official_dir)):
        pytest.skip("checks.flat_episodes has not landed")
    for ep in X.NO_CLIP_ALL_ZERO:
        assert rows[ep]["Hook"] == "No clip: the AI scored every line 0, even when asked again."
        assert list(rows[ep]["Difficulty"]) == ["not scored"]
    for ep in X.NO_CLIP_FLAT:
        assert list(rows[ep]["Difficulty"]) == ["not scored"]


def test_review_stays_with_its_run(app_test, official_dir):
    """Check 15: approve #3 on the official run, switch to first-version and back."""
    at = app_test.run()
    pid = _pid(official_dir, 3)
    at.session_state["open_pid"] = pid
    at.run()
    at.button(key=f"dr-approve-official-{pid}").click().run()
    del at.session_state["open_pid"]
    at.run()
    assert _table(at)["Your call"].iloc[2] == "Approved"
    if find_evidence("previous_version") is None:
        pytest.skip("no first-version run in the bundle")
    at.radio(key="run").set_value("first-version").run()
    assert not at.exception
    assert "Approved" not in _table(at)["Your call"].tolist()
    assert "Evidence runs are read-only" in rendered_text(at)
    at.radio(key="run").set_value("official").run()
    assert _table(at)["Your call"].iloc[2] == "Approved"
    assert "Your review: 1 of 20 decided · 1 approved · 0 to trim · 0 rejected" in rendered_text(at)


def test_export_popover(app_test):
    at = app_test.run()
    text = rendered_text(at)
    for label in ("Cut list (CSV, one row per segment)", "ffmpeg script (cut_promos.sh)", "Readable slate (promos.md)",
                  "Best clip in each episode (CSV)", "Your review (CSV)", "Copy all timecodes"):
        assert label in text
    assert at.toggle(key="exp_only_kept").disabled


@pytest.mark.parametrize("key", ["first-version", "keyword-rules", "consistency-a"])
def test_other_runs_render(app_test, key):
    """Runs without checks, without a first-time viewer or without parts still draw cleanly."""
    from app.ui import data
    if key not in {r.key for r in data.list_runs()}:
        pytest.skip(f"no {key} run in the bundle")
    at = app_test.run()
    at.radio(key="run").set_value(key).run()
    assert not at.exception
    text = rendered_text(at)
    assert "Open #1" in text
    if key == "first-version":
        assert "Not measured: this run predates the editor's checks" in text
        assert "This run predates the editor's checks." in text
    at.button_group(key="promos_list").set_value("episodes").run()
    assert not at.exception


def _render_script(promos_json: str, episodes: int) -> None:
    """Run in AppTest.from_function: the Promos view on a crafted run."""
    import json as _json
    import sys as _sys
    from pathlib import Path as _Path
    root = _Path.cwd()
    if str(root) not in _sys.path:
        _sys.path.insert(0, str(root))
    from app.ui.data import RunInfo
    from app.views import promos as view
    run = RunInfo(key="local-x", dir=root / "no-such-run", role="local", label="Local", caption="", short="Local",
                  show_title="Sample Show", run_id="20261108-100000-x", model="none", episodes=episodes, order=0)
    d = {"promos": _json.loads(promos_json), "metrics": {"episodes": episodes}, "cfg": {}, "run_log": {}, "arc": {},
         "beats": {}, "parse": {}, "promoability": {}, "stability": {}, "benchmark": {}, "per_episode_csv": "",
         "promos_csv": "", "promos_md": "", "cut_script": "", "episodes_list": list(range(1, episodes + 1))}
    view.render(run, d, [run])


def _crafted(promos: list[dict], episodes: int):
    import json as _json
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_render_script, args=(_json.dumps(promos), episodes), default_timeout=60)
    return at.run()


def test_zero_promos_state(official_dir):
    at = _crafted([], 2)
    assert not at.exception
    text = rendered_text(at)
    assert "0 promos" in text and "No clip fits the rules" in text
    assert "See the input check" in text
    assert "0/0" not in text and "None" not in text


def test_small_show_and_escaping(official_dir):
    p = json.loads((official_dir / "promos.json").read_text(encoding="utf-8"))[0]
    p = json.loads(json.dumps(p))
    p["hook"]["text"] = "</script><b>x</b>"
    p.pop("rubric", None)
    p["cold_viewer"] = None
    at = _crafted([p], 2)
    assert not at.exception
    text = rendered_text(at)
    assert "With 2 episodes the opening, middle and late split is coarse" in text
    assert "Not measured: this run predates the editor's checks" in text
    df = _table(at)
    assert df["Hook (first 3 s)"].iloc[0].endswith("“</script><b>x</b>”")   # a dataframe cell is text, never HTML
    node = next(n for n in at.dataframe if "#" in n.value.columns)
    order = list(node.proto.column_order)
    assert "Checks" not in order and "To check" not in order and "Part" in order
    assert "This run predates the editor's checks." in text
    assert "Why it works (template, no AI)" in str(node.proto)
