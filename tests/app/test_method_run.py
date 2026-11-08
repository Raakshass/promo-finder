"""Lane M: Method view, Run it, build_env and the glossary's run-free wording (checks 25 literal scan, 34, 35, 36, 38).

Use the fixtures official_dir and app_test (conftest.py), helpers from apphelpers.py and literal values from
expected_official.py. Derive numbers from the run files where you can."""
from __future__ import annotations

import ast
import io
import re
import tokenize
import zipfile
from pathlib import Path

import pytest
import yaml

import expected_official as X
from apphelpers import ROOT, rendered_text

from app.ui import glossary
from app.views import method, run_it

FORBIDDEN = {"None": re.compile(r"\bNone\b"), "nan": re.compile(r"\bnan\b"),
             "0/0": re.compile(r"(?<![\d.])0/0(?![\d.])"), "dict literal": re.compile(r"\{'")}


def _text(at) -> str:
    """rendered_text with the protobuf text-format quote escapes undone."""
    return rendered_text(at).replace("\\'", "'").replace('\\"', '"')


def _clean(text: str) -> list[str]:
    return [f"{k}: …{text[max(0, m.start() - 40):m.end() + 40]}…" for k, rx in FORBIDDEN.items() for m in rx.finditer(text)]


# ------------------------------------------------------------------------------------------------ Method (check 34)
def _method(app_test, run: str | None = None):
    app_test.query_params["view"] = "Method"
    if run:
        app_test.query_params["run"] = run
    at = app_test.run()
    assert not at.exception, at.exception
    return at


def test_method_official(app_test, official_dir):
    at = _method(app_test)
    text = _text(at)
    assert [h.value for h in at.header] == [method.TITLE]
    for snippet in X.METHOD_SNIPPETS + X.PAID_ESTIMATES:
        assert snippet in text, snippet
    frames = {tuple(df.value.columns): df.value for df in at.dataframe}
    assumptions = frames[("Assumption", "How it was tested", "Result", "Setting to change if wrong")]
    assert len(assumptions) == X.ASSUMPTION_ROWS
    gloss = frames[("Term", "Meaning")]
    assert list(gloss["Term"]) == list(glossary.GLOSSARY)
    assert [s.value for s in at.subheader][:2] == ["Cost and time at volume", "Swap the model"]
    assert not _clean(text), _clean(text)


def test_method_columns_and_learning(app_test, official_dir):
    text = _text(_method(app_test))
    for snippet in X.METHOD_COLUMN_SNIPPETS + (X.CONSISTENCY_MINUTES, "The AI never writes a timestamp"):
        assert snippet in text, snippet
    # Learning from posted promos: the facts from promo_finder/learn.py and the README, in plain words.
    from promo_finder import learn

    assert "Learning from posted promos" in text
    assert "hand-set priors" in text
    assert "promo-finder learn <run> --outcomes posted.csv" in text
    assert f"{int(learn.DEFAULT_PRIOR_STRENGTH)} posts' worth of evidence" in text
    assert "--plan K" in text and "Thompson sampling" in text
    assert "simulated outcomes only" in text and X.LEARN_RECOVERY_POSTS in text and "has not seen real posts yet" in text


def test_method_cost_numbers_follow_prices(official_dir):
    """The estimates are computed from promo_finder.llm.base.PRICES and the run's tokens, not typed in."""
    import json

    from promo_finder.llm.base import PRICES

    llm = json.loads((official_dir / "eval" / "metrics.json").read_text())["llm"]
    for (label, key), expected in zip(method.PAID_MODELS, X.PAID_ESTIMATES):
        cin, cout = PRICES[key]
        assert method._usd(llm["input_tokens"] / 1e6 * cin + llm["output_tokens"] / 1e6 * cout) == expected, label


def test_method_keyword_run(app_test, official_dir):
    at = _method(app_test, "keyword-rules")
    text = _text(at)
    assert "No AI in this run: keyword rules scored every line (0 requests)." in text
    assert "no AI requests" in text
    assert "$0.43" not in text  # no tokens, so no paid-model table
    assert not _clean(text), _clean(text)


def test_raw_file_buttons(app_test, official_dir):
    at = _method(app_test)
    labels = [b.label for b in at.get("download_button")]
    present = [label for label, rel in method.RAW_FILES if (official_dir / rel).exists()]
    assert present and all(label in labels for label in present)


# ------------------------------------------------------------------------------------------------ glossary (check 25)
def test_glossary_has_no_run_numbers(official_dir):
    """The glossary is shown for every run, so it holds no number that belongs to one run (read from the official
    run's own files here, so a new official run needs no edit)."""
    import json

    from app.ui import fmt

    m = json.loads((official_dir / "eval" / "metrics.json").read_text())
    blocks = m["editor_rubric"]["slate_blocks"]
    stab = json.loads((official_dir / "eval" / "stability.json").read_text()) if (official_dir / "eval" / "stability.json").exists() else {}
    run_numbers = [fmt.num(m["windows"]), " / ".join(fmt.num(blocks[k]) for k in ("opening", "middle", "late")),
                   f"{fmt.num(m['finalists'])}-clip", f"{fmt.num(m['finalists'])} best", " here"]
    if isinstance(stab.get("reps"), int):
        run_numbers.append(f"{fmt.num(stab['reps'])} re-runs")
    text = " ".join(glossary.GLOSSARY.values())
    for literal in run_numbers:
        assert literal not in text, literal


def _literal_values() -> tuple[set[float], list[str]]:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    shares = [float(v) for v in cfg["slate"]["shares"].values()]
    variety = [float(cfg["scoring"]["same_episode_penalty"]), float(cfg["scoring"]["same_type_penalty"])]
    penalties = [float(v) for v in cfg["rubric"]["penalties"].values()] + variety + [round(1 - v, 2) for v in variety]
    # Bare numbers in code: the shares and the variety rule's penalties and multipliers. One-decimal values (0.1, 0.9)
    # and the rubric penalties are also ordinary cut-offs, tolerances and chart stops, so they are only looked for in
    # the text forms below ("×0.75"), where they can only mean a penalty.
    numbers = {v for v in set(shares + variety + [round(1 - v, 2) for v in variety]) if round(v, 1) != v}
    texts = sorted({f"×{v:.2f}" for v in penalties} | {f"×{v:g}" for v in penalties}
                   | {f"{round(s * 100)}%" for s in shares})
    return numbers, texts


def test_no_share_or_penalty_literals_in_app():
    """Check 25, literal half: no share or penalty from config.yaml is typed into app/.

    app/ui/glossary.py is excluded on purpose. Its definitions spell out the score formula and the variety rule (the
    weights and the per-repeat multipliers) because those numbers are what the terms mean, and the spec keeps them.
    It holds no number that belongs to one run; test_glossary_has_no_run_numbers checks that."""
    numbers, texts = _literal_values()
    hits: list[str] = []
    for path in sorted((ROOT / "app").rglob("*.py")):
        if path.name == "glossary.py":
            continue
        src = path.read_text(encoding="utf-8")
        docs = {n.body[0].value.lineno for n in ast.walk(ast.parse(src))
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant)
                and isinstance(n.body[0].value.value, str)}
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            where = f"{path.relative_to(ROOT)}:{tok.start[0]}"
            if tok.type == tokenize.STRING and tok.start[0] in docs:
                continue  # docstrings document formatters with examples; they never reach the screen
            if tok.type == tokenize.NUMBER:
                try:
                    if float(tok.string) in numbers:
                        hits.append(f"{where} {tok.string}")
                except ValueError:
                    pass
            elif tok.type == tokenize.STRING:
                for t in texts:
                    if re.search(rf"(?<![\d.]){re.escape(t)}", tok.string):
                        hits.append(f"{where} {t}")
    assert not hits, hits


# ------------------------------------------------------------------------------------------------ build_env (check 35)
SERVER = {"NVIDIA_API_KEY": "srv-nv", "OPENROUTER_API_KEY": "srv-or", "OPENROUTER_API_KEY2": "srv-or2",
          "OPENROUTER_API_KEY3": "srv-or3"}


@pytest.fixture
def dirty_env(monkeypatch):
    """The deployment's secrets also arrive as environment variables."""
    for k, v in {**SERVER, "OPENROUTER_API_KEY_2": "srv-or-u2", "GEMINI_API_KEY": "srv-gem", "GOOGLE_API_KEY": "srv-goog",
                 "PROMO_API_KEY": "srv-promo", "PROMO_BASE_URL": "https://srv.example/v1", "OPENAI_BASE_URL": "x",
                 "SOME_AUTH_TOKEN": "tok", "api_key": "lower", "KEEP_ME": "1"}.items():
        monkeypatch.setenv(k, v)


def _keys(env: dict) -> dict:
    return {k: v for k, v in env.items() if re.search(r"(?:^|_)(?:API_KEY|AUTH_TOKEN|BASE_URL)", k, re.I)}


def test_build_env_free_nvidia(dirty_env):
    env = run_it.build_env("free", SERVER, "pasted", run_it.NVIDIA_DEMO_MODEL, None)
    assert _keys(env) == {"NVIDIA_API_KEY": "srv-nv"}
    assert env["KEEP_ME"] == "1" and env["PYTHON_DOTENV_DISABLED"] == "1"


def test_build_env_free_openrouter(dirty_env):
    env = run_it.build_env("free", SERVER, None, run_it.OPENROUTER_DEMO_MODEL, "https://evil.example/v1")
    assert _keys(env) == {"OPENROUTER_API_KEY": "srv-or", "OPENROUTER_API_KEY2": "srv-or2",
                          "OPENROUTER_API_KEY3": "srv-or3"}


def test_build_env_own_key(dirty_env):
    env = run_it.build_env("own", SERVER, "  mine  ", "gemini/gemini-2.5-flash", "https://ignored.example/v1")
    assert _keys(env) == {"GEMINI_API_KEY": "mine"}
    env = run_it.build_env("own", SERVER, "mine", "openrouter/some/model", None)
    assert _keys(env) == {"OPENROUTER_API_KEY": "mine"}   # never the server's rotation keys
    env = run_it.build_env("own", SERVER, None, "openrouter/some/model", None)
    assert _keys(env) == {}


def test_build_env_custom_endpoint(dirty_env):
    env = run_it.build_env("own", SERVER, "mine", "custom/my-model", " https://kuku.example/v1 ")
    assert _keys(env) == {"PROMO_API_KEY": "mine", "PROMO_BASE_URL": "https://kuku.example/v1"}


def test_build_env_keyword_rules(dirty_env):
    for mode, model in (("none", "none"), ("free", "none"), ("own", "none"), ("own", "not a model")):
        assert _keys(run_it.build_env(mode, SERVER, "mine", model, "https://x.example/v1")) == {}, (mode, model)


# ------------------------------------------------------------------------------------------------ errors (check 38)
@pytest.mark.parametrize("log, msg, icon", [
    ("x: every key has used its free daily\n quota (429). It resets", run_it.QUOTA_MSG, ":material/hourglass_empty:"),
    ("error: HTTP 401 Unauthorized", run_it.KEY_MSG, ":material/key_off:"),
    ("openrouter: HTTP\n403 forbidden", run_it.KEY_MSG, ":material/key_off:"),
    ("no API key for groq: set GROQ_API_KEY in .env", run_it.KEY_MSG, ":material/key_off:"),
    ("Traceback (most recent call last):\nValueError: boom", run_it.OTHER_MSG, ":material/error:"),
])
def test_failure_message(log, msg, icon):
    assert run_it.failure_message(log) == (msg, icon)


def test_failure_wording_matches_spec():
    assert "resets at 05:30 IST" in run_it.QUOTA_MSG and "keyword rules" in run_it.QUOTA_MSG
    assert "twin" not in run_it.QUOTA_MSG
    assert run_it.KEY_MSG == "The provider rejected the key, or no key was given. Check the model name and the key."
    assert run_it.OTHER_MSG == "The run stopped with an error; the last lines of its log are below."


def test_stage_labels():
    log = ["sample-show: 2 episodes · model: none (heuristic twin) · judge: none · run x",
           "aliases merged: 4 labels → 3 canonical names", "show arc (llm): leads …",
           "candidate windows: 9 → finalists: 3", "invariants 100% · cliff localisation 50%"]
    stages = [run_it.STAGES[run_it.stage_of(line)][1] for line in log]
    assert stages == ["Reading the scripts", "Merging speaker names", "Reading the whole show",
                      "Building clips and running the first-time-viewer test", "Writing the results"]
    assert run_it.status_label(3, 12.7) == "Building clips and running the first-time-viewer test · 12 s"


# ------------------------------------------------------------------------------------------------ Run it view
def test_storage_caption_is_honest():
    src = (ROOT / "app" / "views" / "run_it.py").read_text(encoding="utf-8")
    assert "anyone using this demo can see them" in run_it.STORAGE_CAPTION
    assert "?space=" not in src.split('"""', 2)[2] and "only someone with" not in src


def test_sample_zip_and_fixtures():
    with zipfile.ZipFile(io.BytesIO(run_it.sample_zip())) as z:
        assert sorted(Path(n).name for n in z.namelist()) == sorted(run_it.SAMPLE_FILES)


def test_lock(tmp_path, monkeypatch):
    import os
    import time

    monkeypatch.setattr(run_it, "LOCK", tmp_path / "run.lock")
    assert run_it._take_lock()
    assert not run_it._take_lock()
    old = time.time() - run_it.LOCK_S - 5
    os.utime(run_it.LOCK, (old, old))
    assert run_it._take_lock()   # a stale lock is replaced


def _run_it_alone(root: str):
    import sys

    if root not in sys.path:
        sys.path.insert(0, root)
    from app.ui import data
    from app.views import run_it

    run_it.render(None, {}, data.list_runs())


def test_run_it_without_runs(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from app.ui import data

    monkeypatch.setattr(data, "OUT", tmp_path)
    at = AppTest.from_function(_run_it_alone, args=(str(ROOT),), default_timeout=60).run()
    assert not at.exception, at.exception
    go = at.button(key="run_go")
    assert go.disabled and go.help == "Upload scripts or try the sample first."
    assert at.radio(key="run_mode").value == "none"   # no free key in the test environment
    assert not _clean(rendered_text(at))


def test_run_it_sample_check(app_test, official_dir):
    app_test.query_params["view"] = "Run it"
    at = app_test.run()
    at.button(key="run_sample").click().run()
    assert not at.exception, at.exception
    assert at.success[0].value.startswith("Read 2 episodes: ")
    at.radio(key="run_mode").set_value("own").run()
    at.text_input(key="run_own_model").set_value("custom/my-model").run()
    assert at.button(key="run_go").disabled          # custom/ needs a base URL
    at.radio(key="run_mode").set_value("none").run()
    assert not at.button(key="run_go").disabled
    text = rendered_text(at)
    assert run_it.STORAGE_CAPTION in text
    assert not _clean(text)
    import shutil

    shutil.rmtree(at.session_state[run_it.SRC_ROOT], ignore_errors=True)  # the copied sample (no run deletes it here)


def test_run_it_sample_end_to_end(tmp_path, monkeypatch):
    """Check 36 without a browser: sample + keyword rules + Find promos runs the CLI (no model calls), writes the run
    under OUT, and asks the app to open it in Promos. OUT and the lock point at a temp folder."""
    from streamlit.testing.v1 import AppTest

    from app.ui import data

    monkeypatch.setattr(data, "OUT", tmp_path / "out")
    monkeypatch.setattr(run_it, "LOCK", tmp_path / "run.lock")
    at = AppTest.from_function(_run_it_alone, args=(str(ROOT),), default_timeout=180).run()
    at.button(key="run_sample").click().run()
    at.radio(key="run_mode").set_value("none").run()
    at.button(key="run_go").click().run()
    assert not at.exception, at.exception
    made = list((tmp_path / "out").glob("sample-show/*-upload/promos.json"))
    assert len(made) == 1
    assert at.session_state["_goto_view"] == "Promos"
    key = at.session_state["_goto_run"]
    runs = {r.key: r for r in data.list_runs()}
    assert key in runs and runs[key].role == "upload"
    assert runs[key].label.startswith("Your run · Sample Show · 2 episodes · No AI (keyword rules)")
    assert not (tmp_path / "run.lock").exists()
    assert run_it.SRC not in at.session_state     # the copied sample was deleted after the run


# ------------------------------------------------------------------------------------------------ Try learn (optional)
def _official_info():
    from app.ui import data

    return next(r for r in data.list_runs() if r.role == "official")


def test_try_learn_on_simulated_results(official_dir):
    """The Method expander runs promo_finder.learn in-process: old weights from the run's config, new ones fitted."""
    import csv
    import json
    import random

    from app.ui import fmt

    cfg = yaml.safe_load((official_dir / "config.yaml").read_text(encoding="utf-8"))
    promos = json.loads((official_dir / "promos.json").read_text(encoding="utf-8"))
    rng = random.Random(5)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["promo_id", "hook_rate"])
    for p in promos:  # a metric that rises with the run's own score, plus noise
        w.writerow([p["promo_id"], f"{0.1 + 0.5 * p['score'] + rng.gauss(0, 0.02):.4f}"])
    out = method._learn_result(_official_info(), buf.getvalue().encode())
    assert "error" not in out, out.get("error")
    assert out["n"] == len(promos) and out["metric"] == "hook_rate" and out["kind"] == "continuous"
    assert [r["Weight"] for r in out["rows"]] == ["hook", "cliffhanger", "makes sense alone", "pace"]
    assert [r["Now"] for r in out["rows"]] == [fmt.score(cfg["scoring"][k]) for k in
                                               ("w_hook", "w_cliff", "w_standalone", "w_density")]
    learned = [float(r["Learned"]) for r in out["rows"]]
    assert abs(sum(learned) - 1) < 0.01 and all(x >= 0 for x in learned)
    assert any("posts" in n for n in out["notes"])  # fewer than 30 posts: the module warns that it is mostly the prior


def test_try_learn_bad_files(official_dir):
    run = _official_info()
    assert method._learn_result(run, b"name,hook_rate\nx,0.3\n")["error"].startswith("Nothing could be learned")
    assert "error" in method._learn_result(run, b"promo_id,hook_rate\nP01,abc\n")
    assert "error" in method._learn_result(run, b"\xff\xfe\x00garbage")
