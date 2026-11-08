"""Lane R: checks.py and the Quality view (checks 30-33).

Use the fixtures official_dir and app_test (conftest.py), helpers from apphelpers.py and literal values from
expected_official.py. Derive numbers from the run files where you can."""
from __future__ import annotations

import re

import expected_official as X
import pytest
from apphelpers import find_evidence, rendered_text
from test_hygiene import ALLOWED_PHRASES, FORBIDDEN

from app.ui import checks
from app.ui import data as D


@pytest.fixture(scope="module")
def run(official_dir):
    return D.load_run(str(official_dir), D.stamp(official_dir))


def _dirty(text: str) -> list[str]:
    """Hygiene hits (check 40), with the same patterns as test_hygiene.py."""
    for phrase in ALLOWED_PHRASES:
        text = text.replace(phrase, "")
    return [f"{name}: {text[max(0, m.start() - 40):m.end() + 40]}" for name, rx in FORBIDDEN.items()
            for m in rx.finditer(text)]


def _quality(at, run_key: str | None = None):
    at.session_state["view"] = "Quality"
    if run_key:
        at.query_params["run"] = run_key
    at.run()
    assert not at.exception
    return rendered_text(at)


# ---------------------------------------------------------------- pure functions

def test_problems_official(run):
    shown, missing = checks.problems(run)
    assert len(shown) == X.N_PROBLEMS and not missing
    text = "\n".join(f"{p.title} {p.body}" for p in shown)
    for s in X.PROBLEM_SNIPPETS_V2:
        assert s in text, s
    # the old 2-drama claim is gone from the problems list
    assert "0.91" not in text and "no better than chance" not in text
    assert shown[0].source.startswith("docs/research.md")
    assert all(p.source for p in shown)


def test_problem_order_follows_spec(run):
    shown, _ = checks.problems(run)
    keys = ["professional editors' ads", "Two runs agree", "fail at least one editor's check", "answers flat",
            "misses the writers' own cliffhanger", "Keyword rules find", "blind judge", "weights"]
    for p, k in zip(shown, keys):
        assert k in f"{p.title} {p.body}", (k, p.title)


def test_problems_drop_missing_files(run):
    d = dict(run, stability={}, ab={}, benchmark={})
    shown, missing = checks.problems(d)
    assert len(shown) == X.N_PROBLEMS - 3 and len(missing) == 3


def test_measured_checks(run):
    rows = checks.measured_checks(run)
    assert all(set(r) == {"What was checked", "Result", "Compared with", "How to read it", "Source"} for r in rows)
    text = "\n".join(" ".join(r.values()) for r in rows)
    for s in X.MEASURED_SNIPPETS:
        assert s in text, s
    assert all(r["Source"] for r in rows)
    assert "None" not in text and "nan" not in text


def test_flat_episodes(run):
    rows = checks.flat_episodes(run)
    assert len(rows) == run["metrics"]["answer_repairs"]["flat_answers"]
    assert tuple(r["episode"] for r in rows if r["all_zero"]) == X.NO_CLIP_ALL_ZERO
    assert tuple(r["episode"] for r in rows if r["still_flat"] and not r["all_zero"]) == X.NO_CLIP_FLAT


def test_fail_count(run):
    assert checks.fail_count(run) == X.FAIL_COUNT
    assert checks.fail_count(dict(run, promos=[])) == 0


# ---------------------------------------------------------------- checks 30, 31, 33 (AppTest)

def test_quality_view_official(app_test):
    text = _quality(app_test)
    assert "Where it picks badly" in text
    nums = [m.value for m in app_test.markdown if re.match(r"^\d+\\\. \*\*", m.value)]
    assert len(nums) == X.N_PROBLEMS
    for s in X.PROBLEM_SNIPPETS_V2 + X.MEASURED_SNIPPETS:
        assert s in text, s
    assert X.PROMISE_CASE_LINE in text
    labels = [e.label for e in app_test.expander]
    for lab in X.CASE_LABELS:
        assert lab in labels, lab
    assert "Known limits" in text and "Runs behind these numbers" in text
    assert "None is never answered" not in text
    assert not _dirty(text)
    assert X.CUES_HIDDEN_PCT in text and X.OFFICIAL_RECALL_PCT in text


def test_quality_case_tables_have_open_buttons(app_test):
    _quality(app_test)
    frames = [f for f in app_test.dataframe if "#" in f.value.columns]
    assert frames
    for f in frames:
        assert all(re.fullmatch(r"Open #\d+", v) for v in f.value["#"].tolist())


def test_open_from_parses_rank(monkeypatch):
    from app.views import quality

    store = {"q-open-x": {"row": 2, "label": "Open #7"}}
    opened = []
    monkeypatch.setattr(quality.st, "session_state", store)
    monkeypatch.setattr(quality.state, "open_promo", opened.append)
    quality._open_from("q-open-x", {7: "P07"})
    assert opened == ["P07"]


# ---------------------------------------------------------------- check 32 (keyword-rules run) and other runs

def test_quality_keyword_run(app_test):
    if find_evidence("heuristic_only") is None:
        pytest.skip("no keyword-rules run")
    text = _quality(app_test, "keyword-rules")
    assert not _dirty(text)
    assert "Not checked: this run used no AI model" in text
    caps = [c.value for c in app_test.caption if c.value.startswith("Not measured for this run:")]
    assert caps


@pytest.mark.parametrize("key,prefix", [("first-version", "previous_version"), ("consistency-a", "stability_base")])
def test_quality_other_runs(app_test, key, prefix):
    if find_evidence(prefix) is None:
        pytest.skip(f"no {key} run")
    text = _quality(app_test, key)
    assert not _dirty(text)
    assert "Runs behind these numbers" not in text
