"""E0: fmt, glossary, data, state and the frame (gate, header, picker, banner, tabs, footer, drawer hook).
Expected values come from the run's own files where possible; fixed literals live in expected_official.py."""
from __future__ import annotations

import json
import shutil

import pytest

import expected_official as X
from app.ui import data, fmt, glossary
from apphelpers import find_evidence, rendered_text

VIEWS = ["Promos", "Ranking", "Quality", "Method", "Run it"]


# ----------------------------------------------------------------------------- fmt (spec 4.1 examples)

@pytest.mark.parametrize("fn,arg,want", [
    (fmt.score, 0.7583, "0.758"), (fmt.mult, 0.85, "×0.85"), (fmt.pct, 0.4, "40%"), (fmt.pct, 0.913, "91%"),
    (fmt.lift, 1.395, "1.40×"), (fmt.secs, 24.0, "24.0 s"), (fmt.clock, 82.5, "1:22.5"), (fmt.mmss, 90, "1:30"),
    (fmt.minutes, 1176, "19.6 min"), (fmt.ordinal, 3, "3rd"), (fmt.ordinal, 11, "11th"), (fmt.ordinal, 22, "22nd"),
    (fmt.show_title, "i-married-a-broken-soldier", "I Married a Broken Soldier"),
    (fmt.show_title, "the-end-of-it", "The End of It"),
    (fmt.run_time, "20261107-182015-openrouter-x", "7 Nov 2026, 18:20"), (fmt.run_time, "not-a-run", ""),
    (fmt.run_time_short, "20261107-182015-x", "7 Nov 18:20"), (fmt.run_date, "20261106-025131-x", "6 Nov"),
    (fmt.model_short, "nvidia/nemotron-3.5-super-vl:free", "Nemotron 3.5 Super"),
    (fmt.model_short, "none", "No AI (keyword rules)"), (fmt.model_short, "heuristic", "No AI (keyword rules)"),
    (fmt.model_short, "mlx-community/Llama-3.1-8B-Instruct-4bit", "Llama 3.1 8B (local)"),
    (fmt.model_short, "mlx-community/Qwen3-4B-Instruct-2507-4bit", "Qwen3 4B (local)"),
    (fmt.model_short, "acme/house-model-7b:free", "house-model-7b"),
    (fmt.num, 158057, "158,057"),
])
def test_fmt_examples(fn, arg, want):
    assert fn(arg) == want


def test_fmt_multi_arg():
    assert fmt.k_of_n(16, 24) == "16 of 24"
    assert fmt.ranges([1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 24]) == "1–7, 9–11, 24"
    assert fmt.ranges([]) == ""
    assert fmt.ep_range(1, 10) == "episodes 1–10" and fmt.ep_range(4, 4) == "episode 4"
    s = {"episode": 10, "start": "01:22.50", "end": "01:34.00"}
    assert fmt.seg(s) == "Ep 10 01:22.50–01:34.00"
    assert fmt.segs({"segments": [s, {"episode": 11, "start": "00:00.00", "end": "00:20.00"}]}) == X.SEGMENTS_OF_5
    assert fmt.said("CAPTAIN", "Hold the gate!") == "CAPTAIN: “Hold the gate!”"
    assert fmt.said(None, "She slaps him.") == "On screen: She slaps him."
    assert fmt.plural(1, "segment") == "1 segment" and fmt.plural(2, "segment") == "2 segments"
    assert fmt.is_free("nvidia/nemotron-3.5-super-vl:free") and fmt.is_free("x", provider="nvidia")
    assert not fmt.is_free("house-model-7b", provider="acme")


def test_escapes():
    assert "<" not in fmt.esc("</script>") and fmt.esc(None) == ""
    assert fmt.md_esc("*bold* _x_ [a](b) `c` $x$") == r"\*bold\* \_x\_ \[a\](b) \`c\` \$x\$"


# ----------------------------------------------------------------------------- glossary

def test_checks_match_rubric_labels():
    from promo_finder.rubric import LABELS

    assert list(glossary.CHECKS) == list(LABELS)
    assert all(c.key == k for k, c in glossary.CHECKS.items())


def test_glossary_terms():
    assert len(glossary.GLOSSARY) == 50
    assert glossary.term("Hook").startswith("The opening line")
    with pytest.raises(KeyError):
        glossary.term("No such term")
    assert glossary.penalty({"rubric": {"penalties": {"small_cast": 0.9}}}, "small_cast") == 0.9
    assert glossary.penalty({}, "small_cast") == 1.0


def test_penalties_read_from_run_config(official_dir):
    import yaml

    cfg = yaml.safe_load((official_dir / "config.yaml").read_text())
    for key in glossary.CHECKS:
        assert glossary.penalty(cfg, key) == float(cfg["rubric"]["penalties"].get(key, 1.0))


# ----------------------------------------------------------------------------- data

LOAD_KEYS = {"promos", "metrics", "cfg", "run_log", "arc", "beats", "parse", "aliases", "promoability", "stability", "votes",
             "ab", "benchmark", "dramaad", "assumptions", "check_exp", "check_exp_ends_open", "promos_csv", "promos_md",
             "per_episode_csv", "cut_script", "failures_md", "official", "episodes_list", "files"}


def test_out_points_at_bundle():
    assert data.OUT.name == "outputs" and data.OUT.parent.name == "submission"


def test_list_runs_official_first(official_dir):
    runs = data.list_runs(None)
    assert runs[0].key == "official" and runs[0].role == "official"
    assert runs[0].dir.resolve() == official_dir.resolve()
    metrics = json.loads((official_dir / "eval" / "metrics.json").read_text())
    assert runs[0].run_id == metrics["run_id"] and runs[0].episodes == metrics["episodes"]
    assert [r.order for r in runs] == list(range(len(runs)))
    assert len({r.key for r in runs}) == len(runs)
    assert runs[0].label == "Official results" and runs[0].short == "Official"


def test_list_runs_evidence_keys_and_order(official_dir):
    want = [(p, k) for p, k in (("previous_version", "first-version"), ("heuristic_only", "keyword-rules"),
                                ("cue_blind_ablation", "cues-hidden"), ("stability_base", "consistency-a"),
                                ("stability_rerun", "consistency-b")) if find_evidence(p)]
    runs = data.list_runs(None)
    evid = [r for r in runs if r.role == "evidence"]
    assert [r.key for r in evid] == [k for _, k in want]
    kw = next((r for r in evid if r.key == "keyword-rules"), None)
    if kw:
        assert kw.model == "none" and "__" not in kw.run_id


def test_cues_hidden_caption_reads_both_runs(official_dir):
    cue = find_evidence("cue_blind_ablation")
    if cue is None:
        pytest.skip("no cues-hidden run")
    pct = lambda d: fmt.pct(json.loads((d / "eval" / "metrics.json").read_text())["cliff_localisation"]["recall_all"])  # noqa: E731
    run = next(r for r in data.list_runs(None) if r.key == "cues-hidden")
    assert f"{pct(cue)} of episodes ({pct(official_dir)} with them)" in run.caption


def test_load_run_keys_and_types(official_dir):
    d = data.load_run(str(official_dir), data.stamp(official_dir))
    assert set(d) == LOAD_KEYS
    promos = json.loads((official_dir / "promos.json").read_text())
    assert len(d["promos"]) == len(promos) and d["official"] is True
    assert d["episodes_list"] == sorted(d["beats"]) and len(d["beats"]) == len(list((official_dir / "beats").glob("ep-*.json")))
    assert "promos.json" in d["files"] and "eval/metrics.json" in d["files"]
    assert isinstance(d["cfg"], dict) and d["cfg"].get("rubric")


def test_load_run_missing_and_corrupt(tmp_path):
    run = tmp_path / "show" / "20260101-000000-heuristic-x"
    run.mkdir(parents=True)
    (run / "promos.json").write_text("[]")
    d = data.load_run(str(run), data.stamp(run))
    assert set(d) == LOAD_KEYS and d["promos"] == [] and d["metrics"] == {} and d["promos_csv"] == "" and d["beats"] == {}
    (run / "eval").mkdir()
    (run / "eval" / "metrics.json").write_text("{not json")
    with pytest.raises(data.RunReadError) as e:
        data.load_run(str(run), data.stamp(run) + 1)
    assert e.value.file == "eval/metrics.json"


def test_local_and_upload_runs_listed(tmp_path, monkeypatch, official_dir):
    show = tmp_path / "my-show"
    for rid in ("20261108-101500-heuristic-mytag", "20261108-111500-heuristic-upload"):
        (show / rid).mkdir(parents=True)
        (show / rid / "promos.json").write_text("[]")
    dup = tmp_path / "i-married-a-broken-soldier" / official_dir.name.split("__", 1)[-1]  # same run_id, no OFFICIAL
    shutil.copytree(official_dir, dup, ignore=shutil.ignore_patterns("OFFICIAL", "llm_raw", "beats", "candidates.jsonl"))
    monkeypatch.setattr(data, "OUT", tmp_path)
    runs = data.list_runs(None)
    assert [r.role for r in runs] == ["upload", "local", "local"]
    up, local = runs[0], runs[1]
    assert up.key == "upload-111500" and up.label.startswith("Your run · My Show · ") and up.short == "Your run 11:15"
    assert local.key == "local-20261108-101500-heuristic-mytag" and local.label.endswith(" · mytag")
    shutil.copy(official_dir / "OFFICIAL", dup / "OFFICIAL")
    data.list_runs.clear()
    assert [r.key for r in data.list_runs(None)][0] == "official"


def test_fetch_outputs_noop_without_url(monkeypatch):
    monkeypatch.delenv("OUTPUTS_URL", raising=False)
    assert data.fetch_outputs() is None


# ----------------------------------------------------------------------------- state (inside a script run)

def _state_script():
    import streamlit as st

    from app.ui import state

    state.init()
    state.review_set("r1", "P01", "approve")
    state.review_set("r1", "P03", "trim", note="start 2 s later")
    state.review_set("r1", "P04", "reject", reason="Weak hook")
    state.review_set("r1", "P04", "approve")
    state.review_set("r2", "P01", "reject")
    state.review_set("r2", "P01", None)
    st.session_state["out"] = {
        "get": state.review_get("r1", "P03"), "none": state.review_get("r1", "P99"),
        "counts": state.review_counts("r1", ["P01", "P02", "P03", "P04"]), "r2": state.review_all("r2"),
        "p04": state.review_get("r1", "P04"),
    }


def test_review_store():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_state_script, default_timeout=60)
    at.run()
    out = at.session_state["out"]
    assert out["get"]["call"] == "trim" and out["get"]["note"] == "start 2 s later" and "T" in out["get"]["at"]
    assert out["none"] is None and out["r2"] == {}
    assert out["counts"] == {"decided": 3, "approve": 2, "trim": 1, "reject": 0, "open": 1}
    assert out["p04"]["reason"] == ""  # the reason only belongs to a rejection


# ----------------------------------------------------------------------------- the frame (AppTest)

def test_app_smoke(app_test, official_dir):
    at = app_test.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == VIEWS
    assert len(at.sidebar.children) == 0
    caps = [c.value for c in at.caption]
    assert X.RUN_LINE in caps
    assert any(fmt.show_title(official_dir.parent.name) in h.value for h in at.html)
    n = len(json.loads((official_dir / "promos.json").read_text()))
    assert any(f"{n} promos" in c for c in caps)  # the data path reaches the Promos view
    assert not at.warning  # official run: no banner
    assert caps[-1].startswith("Contains excerpts from a confidential script")


def test_view_query_param(app_test):
    app_test.query_params["view"] = "Quality"
    at = app_test.run()
    assert not at.exception
    assert [h.value for h in at.header] == ["Where it picks badly"]


def test_goto_callback_switches_tab(app_test):
    at = app_test.run()
    at.session_state["view"] = "Method"
    at.run()
    assert [h.value for h in at.header] == ["How it works"]


def test_switch_run_banner_and_back(app_test):
    if not find_evidence("heuristic_only"):
        pytest.skip("no keyword-rules run")
    at = app_test.run()
    at.radio(key="run").set_value("keyword-rules").run()
    assert not at.exception and at.session_state["run"] == "keyword-rules"
    assert at.query_params["run"] == "keyword-rules"
    assert any("evidence run" in w.value for w in at.warning)
    at.button(key="banner-back").click().run()
    assert at.session_state["run"] == "official" and not at.warning
    assert "run" not in at.query_params


def test_run_query_param(app_test):
    if not find_evidence("previous_version"):
        pytest.skip("no first-version run")
    app_test.query_params["run"] = "first-version"
    at = app_test.run()
    assert at.session_state["run"] == "first-version" and any("First version" in w.value for w in at.warning)


def test_invalid_run_query_param_falls_back(app_test):
    app_test.query_params["run"] = "upload-lost"
    at = app_test.run()
    assert not at.exception and at.session_state["run"] == "official" and "run" not in at.query_params


def test_drawer_hook(app_test, official_dir):
    first = json.loads((official_dir / "promos.json").read_text())[0]["promo_id"]
    app_test.session_state["open_pid"] = first
    at = app_test.run()
    assert not at.exception and at.session_state["open_pid"] == first
    at.session_state["open_pid"] = "not-a-promo"
    at.run()
    assert "open_pid" not in at.session_state


def test_gate(app_test, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    at = app_test.run()
    assert not at.tabs and len(at.text_input) == 1
    at.text_input[0].input("wrong").run()
    assert at.error and not at.tabs
    at.text_input[0].input("s3cret").run()
    assert not at.exception and [t.label for t in at.tabs] == VIEWS


def test_no_runs(app_test, monkeypatch, tmp_path):
    monkeypatch.setattr(data, "OUT", tmp_path)
    at = app_test.run()
    assert not at.exception and [t.label for t in at.tabs] == VIEWS
    assert "No results to show yet" in rendered_text(at)
    at.session_state["view"] = "Run it"
    at.run()
    assert not at.exception and "Run it on a new show" in [h.value for h in at.header]
