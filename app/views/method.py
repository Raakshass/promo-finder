"""Method view (spec 5.6). Owner: Lane M.

The diagram; what the AI, code and the editor each do; cost and time, with list-price estimates for paid models; how to
swap the model; how the weights can learn from posted promos; the assumptions table; the glossary; the run's raw files.
Every number is read from the selected run's files (or promo_finder's own constants), never typed in.
"""
from __future__ import annotations

import ast
import math
import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from app.ui import data as data_mod
from app.ui import fmt
from app.ui.components import AI, not_measured, view_header
from app.ui.data import RunInfo
from app.ui.glossary import CHECKS, GLOSSARY, PART_NAMES

TITLE = "How it works"
PURPOSE = "What the AI does, what code does, what the editor decides, what a show costs, and the assumptions behind it."

DEFAULT_REPO = "https://github.com/Raakshass/promo-finder"

# Paid models for the list-price estimate: display name -> key in promo_finder.llm.base.PRICES (spec 5.6).
PAID_MODELS = (("Gemini 2.5 Flash", "gemini/gemini-2.5-flash"),
               ("GPT-4.1 mini", "openai/gpt-4.1-mini"),
               ("GPT-4.1", "openai/gpt-4.1"))
SHOWS_AT_VOLUME = 100

# Raw files offered for download: label -> path inside the run folder.
RAW_FILES = (("metrics.json", "eval/metrics.json"), ("failures.md", "eval/failures.md"),
             ("promoability.md", "eval/promoability.md"), ("assumption_checks.md", "eval/assumption_checks.md"),
             ("run_log.json", "run_log.json"), ("config.yaml", "config.yaml"), ("show_map.svg", "show_map.svg"),
             ("promos.md", "promos.md"))
_MIME = {".json": "application/json", ".md": "text/markdown", ".yaml": "text/yaml", ".svg": "image/svg+xml"}

# Model ids whose weights are published (used only for the words "open weights" in the AI column's heading).
_OPEN_WEIGHTS = ("nemotron", "llama", "qwen", "gemma", "deepseek", "gpt-oss")
_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
_TERM_NAMES = {"hook": "hook", "cliff": "cliffhanger", "standalone": "makes sense alone", "density": "pace"}


# ----------------------------------------------------------------------------------------------------------- helpers
def _int(x) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def _float(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _secs(x: float) -> str:
    """20.0 -> "20 s"; 1.5 -> "1.5 s"."""
    return f"{fmt.num(x)} s" if float(x).is_integer() else fmt.secs(x)


def _usd(x: float, places: int = 2) -> str:
    return f"${x:,.{places}f}"


def _word(n: int) -> str:
    return _WORDS.get(int(n), fmt.num(n))


def _stage(llm: dict, name: str) -> int:
    return _int(((llm.get("per_stage") or {}).get(name) or {}).get("calls"))


def _no_model(run: RunInfo, d: dict) -> bool:
    prov = str(((d.get("metrics") or {}).get("model") or {}).get("provider") or "").lower()
    return run.model == "none" or prov == "none"


def _model_name(run: RunInfo, d: dict) -> tuple[str, bool]:
    """(short name, free tier?)."""
    mm = (d.get("metrics") or {}).get("model") or {}
    provider = str(mm.get("provider") or "")
    free = fmt.is_free(run.model, provider) or fmt.is_free(str((d.get("run_log") or {}).get("args", {}).get("model") or ""))
    return fmt.model_short(run.model), free


def _unapplied_merges(aliases: dict) -> int:
    """Speaker-name merges the AI proposed that code left for a person (from aliases.yaml's notes)."""
    notes = str((aliases or {}).get("notes") or "")
    m = re.search(r"not applied \(needs a human\):\s*(\{.*\})", notes, re.S)
    if not m:
        return 0
    try:
        v = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return 0
    return len(v) if isinstance(v, dict) else 0


def _readable(text: object) -> str:
    """Script-written prose can carry Python literals ({'opening': 11, ...}, [8, 14]); show them as plain words."""
    def plain(m: re.Match) -> str:
        try:
            v = ast.literal_eval(m.group(0))
        except (ValueError, SyntaxError):
            return m.group(0)
        if isinstance(v, dict):
            return ", ".join(f"{str(k).replace('_', ' ')} {fmt.num(x) if isinstance(x, int) else x}" for k, x in v.items())
        if isinstance(v, (list, tuple)):
            return ", ".join(fmt.num(x) if isinstance(x, int) else str(x) for x in v)
        return m.group(0)

    return re.sub(r"\{[^{}]*\}|\[[^\[\]]*\]", plain, "" if text is None else str(text))


def _refused_entailment(llm: dict) -> int:
    return sum(1 for e in llm.get("errors") or [] if str(e).startswith("transport/Entail") and "429" in str(e))


# ----------------------------------------------------------------------------------------------------------- sections
def _diagram() -> None:
    svg = data_mod.ROOT / "docs" / "diagram.svg"
    if not svg.exists():
        return
    try:
        body = svg.read_text(encoding="utf-8")
    except OSError:
        return
    body = re.sub(r"<\?xml[^>]*\?>", "", body)
    st.html(f'<div style="max-width:100%;overflow-x:auto">{body}</div>')
    st.caption("The AI judges drama; code owns every timestamp and the order; the editor decides.")


def _ai_column(run: RunInfo, d: dict) -> None:
    m = d.get("metrics") or {}
    llm = m.get("llm") or {}
    if _no_model(run, d):
        st.markdown("**The AI does**")
        st.markdown("No AI in this run: keyword rules scored every line (0 requests).")
        return
    name, free = _model_name(run, d)
    tags = [t for t in (("open weights" if any(k in run.model.lower() for k in _OPEN_WEIGHTS) else ""),
                        ("free tier" if free else "")) if t]
    st.markdown(f"{AI} **The AI does** ({fmt.md_esc(', '.join([name] + tags))})")
    lines: list[str] = []
    eps = _int(m.get("episodes")) or run.episodes
    score_calls = _stage(llm, "score_beats")
    if score_calls:
        flat = _int((m.get("answer_repairs") or {}).get("flat_answers"))
        first = score_calls - flat
        detail = fmt.plural(score_calls, "request")
        if 0 < flat < score_calls:
            detail += f": {fmt.num(first)} for the {fmt.plural(eps, 'episode')}, {fmt.num(flat)} to re-ask flat answers"
        lines.append(f"Scores every line 0–3 as a hook and as a cliffhanger ({detail}).")
    if _stage(llm, "show_arc"):
        lines.append("Reads the writers' episode summaries once for the leads, turning points and reveals "
                     f"({fmt.num(_stage(llm, 'show_arc'))}).")
    if _stage(llm, "aliases"):
        lines.append(f"Proposes speaker-name merges ({fmt.num(_stage(llm, 'aliases'))}).")
    cv_calls, cv_n = _stage(llm, "cold_viewer"), _int((m.get("cold_viewer") or {}).get("n"))
    if cv_calls and cv_n:
        each = f", {fmt.num(cv_n // cv_calls)} clips each" if cv_n % cv_calls == 0 and cv_n > cv_calls else ""
        lines.append(f"Plays a first-time viewer for the {fmt.num(cv_n)} shortlisted clips "
                     f"({fmt.plural(cv_calls, 'request')}{each}).")
    if _stage(llm, "ledger"):
        lines.append(f"Finds where the show answers each promo's question ({fmt.num(_stage(llm, 'ledger'))}).")
    pitched = sum(1 for p in d.get("promos") or [] if str(p.get("post_first_reason") or "").strip())
    if _stage(llm, "rank_reasons") and pitched:
        lines.append(f"Writes a one-line pitch for the top {fmt.num(pitched)} ({fmt.num(_stage(llm, 'rank_reasons'))}).")
    ent, refused = _stage(llm, "entailment"), _refused_entailment(llm)
    if ent or refused:
        detail = f"{fmt.num(ent)} answered" + (f"; {fmt.num(refused)} refused by the daily limit" if refused else "")
        lines.append(f"Checks the writers' cliffhangers against continuity notes ({detail}).")
    if not lines:
        lines.append("Request counts were not recorded for this run.")
    st.markdown("\n".join(f"- {x}" for x in lines))


def _code_column(run: RunInfo, d: dict) -> None:
    m, cfg, parse = d.get("metrics") or {}, d.get("cfg") or {}, d.get("parse") or {}
    promo = cfg.get("promo") or {}
    st.markdown("**Code does**")
    lines: list[str] = []
    blocks = _int(parse.get("total_blocks"))
    lines.append("Reads every timestamp" + (f" ({fmt.num(blocks)} blocks)." if blocks else "."))
    rules = []
    lo, hi = _float(promo.get("min_duration_s")), _float(promo.get("max_duration_s"))
    if lo is not None and hi is not None:
        rules.append(f"{fmt.num(lo) if lo.is_integer() else fmt.secs(lo)[:-2]}–{_secs(hi)}")
    snap = _float(promo.get("cut_snap_s"))
    if snap:
        rules.append(f"starting on the camera cut before the opening line when one is within {_secs(snap)}")
    hook = _float(promo.get("hook_window_s"))
    if hook:
        rules.append(f"hook within {_secs(hook)}")
    rules.append("no line cut in half")
    windows = _int(m.get("windows"))
    lines.append("Builds every clip that fits the rules: " + ", ".join(rules)
                 + (f" ({fmt.num(windows)} candidates)." if windows else "."))
    by_code = sum(1 for c in CHECKS.values() if c.decided_by != "First-time viewer (AI)")
    lines.append(f"Runs {fmt.num(by_code)} of the {fmt.num(len(CHECKS))} editor's checks.")
    lines.append("Scores with fixed weights from config.yaml.")
    lines.append("Drops overlaps.")
    finalists = _int(m.get("finalists"))
    blocks_ = (m.get("editor_rubric") or {}).get("slate_blocks") or {}
    counts = [(_int(blocks_.get(k)), PART_NAMES[k].lower()) for k in PART_NAMES if k in blocks_]
    n_promos = len(d.get("promos") or [])
    if finalists and counts:
        lines.append(f"Shortlists {fmt.num(finalists)}, then fills the " + " / ".join(fmt.num(n) for n, _ in counts)
                     + " places (" + ", ".join(p for _, p in counts) + ").")
    elif finalists:
        lines.append(f"Shortlists {fmt.num(finalists)}, then picks the {fmt.plural(n_promos, 'promo')}.")
    rejected = sum(1 for p in d.get("promos") or [] if (p.get("promise") or {}).get("status") == "rejected")
    if not _no_model(run, d):
        lines.append(f"Rejects impossible AI answers ({fmt.num(rejected)} in this run).")
    lines.append("Writes the cut list and the ffmpeg script.")
    st.markdown("\n".join(f"- {x}" for x in lines))
    st.markdown("**The AI never writes a timestamp: it scores lines by id.**")


def _editor_column(run: RunInfo, d: dict) -> None:
    promos = d.get("promos") or []
    st.markdown("**The editor does**")
    joins = sum(1 for p in promos if len(p.get("segments") or []) > 1)
    scenes = sum(1 for p in promos if (p.get("flags") or {}).get("scene_change"))
    reveals = sum(1 for p in promos if (p.get("flags") or {}).get("major_reveal"))
    lines = ["Approves, trims or rejects each promo."]
    if joins or scenes:
        lines.append(f"Watches the flagged cuts ({fmt.plural(joins, 'join')}, {fmt.plural(scenes, 'scene change')} "
                     "in this run).")
    if reveals:
        lines.append(f"Decides on clips from episodes with a major reveal ({fmt.num(reveals)} in this run).")
    merges = _unapplied_merges(d.get("aliases") or {})
    if merges:
        lines.append(f"Confirms the {fmt.plural(merges, 'name merge')} the AI proposed but code did not apply.")
    st.markdown("\n".join(f"- {x}" for x in lines))


def _seconds_of(r: RunInfo) -> float | None:
    try:
        return _float((data_mod.load_run(str(r.dir), data_mod.stamp(r.dir)).get("metrics") or {}).get("seconds"))
    except data_mod.RunReadError:
        return None


def _cost(run: RunInfo, d: dict, runs: list[RunInfo]) -> None:
    st.subheader("Cost and time at volume", anchor=False)
    m = d.get("metrics") or {}
    llm = m.get("llm") or {}
    seconds = _float(m.get("seconds"))
    took = ""
    if seconds is not None:
        took = f" {fmt.minutes(seconds) if seconds >= 60 else fmt.secs(seconds)} from start to finish."
    tin, tout = _int(llm.get("input_tokens")), _int(llm.get("output_tokens"))
    name, free = _model_name(run, d)
    if _no_model(run, d):
        st.markdown(f"This run: no AI requests, so no tokens and no cost.{took}")
    else:
        total = _int(llm.get("calls_total"))
        refused = _stage(llm, "transport")
        new = max(_int(llm.get("calls_live")) - refused, 0)
        reused = _int(llm.get("calls_cached"))
        parts = [fmt.plural(new, "new answer"), f"{fmt.num(reused)} reused from an earlier identical request"]
        if refused:
            parts.append(f"{fmt.num(refused)} refused by " + ("the free tier's daily limit" if free else "the provider"))
        cost = _float(llm.get("est_cost_usd_at_paid_rates")) or 0.0
        if free:
            money = "$0 on a free open model."
        elif cost > 0:
            money = f"{_usd(cost)} at list prices."
        else:
            money = f"Cost not estimated: no price on file for {fmt.md_esc(name)}."
        st.markdown(f"This run: {fmt.plural(total, 'AI request')}: {', '.join(parts)}. "
                    f"{fmt.num(tin)} tokens in and {fmt.num(tout)} out. {money}{took}")
    by_key = {r.key: r for r in runs}
    a, b = by_key.get("consistency-a"), by_key.get("consistency-b")
    if a and b and a.model == run.model and not _no_model(run, d):
        sa, sb = _seconds_of(a), _seconds_of(b)
        if sa and sb:
            st.markdown(f"Runs of the same model with every answer new took {fmt.minutes(sa)[:-4]} and {fmt.minutes(sb)} "
                        "on the same free tier (consistency runs A and B, an earlier version).")
    if tin or tout:
        from promo_finder.llm.base import PRICES

        rows = []
        for label, key in PAID_MODELS:
            price = PRICES.get(key)
            if not price:
                continue
            per_show = tin / 1e6 * price[0] + tout / 1e6 * price[1]
            rows.append({"Model": label, "Per show": _usd(per_show),
                         f"Per {fmt.num(SHOWS_AT_VOLUME)} shows": _usd(per_show * SHOWS_AT_VOLUME, 0)})
        if rows:
            st.markdown("**If a paid model needed the same tokens** (rough estimate at list prices)")
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, column_order=list(df.columns), width="content")
            st.caption("Another model uses a different number of tokens; treat these as order of magnitude.")


def _swap() -> None:
    st.subheader("Swap the model", anchor=False)
    st.markdown("One setting picks the model. In this app: **Run it → Your own key or endpoint**. On a machine: "
                "`--model provider/model` (or `PROMO_MODEL`).")
    st.markdown("Providers: nvidia (free), openrouter, gemini, groq, mistral, openai, ollama; `custom/<model>` for any "
                "OpenAI-compatible endpoint, such as Kuku's own, with a base URL; `none` for keyword rules.")
    st.code("uv sync && uv run promo-finder run <folder> --model <provider/model>", language="bash")


def _learning(run: RunInfo, d: dict) -> None:
    from promo_finder import learn

    st.subheader("Learning from posted promos", anchor=False)
    scoring = (d.get("cfg") or {}).get("scoring") or {}
    weights = [(_TERM_NAMES[t], _float(scoring.get(f"w_{t}"))) for t in learn.TERMS]
    now = ", ".join(f"{name} {fmt.pct(w)}" for name, w in weights if w is not None)
    lines = [
        "The four ranking weights are hand-set priors, not learned" + (f" (this run: {now})." if now else "."),
        "`promo-finder learn <run> --outcomes posted.csv` refits them from how posted promos did: their hook rate, "
        "their hold rate, or the editor's approve or reject (1 or 0).",
        f"The current weights count as {fmt.num(learn.DEFAULT_PRIOR_STRENGTH)} posts' worth of evidence, so a few posts "
        "cannot swing them.",
        "`--plan K` picks the next K posts by Thompson sampling: it draws plausible weights from the fit, re-ranks the "
        "clips under each draw and posts what wins most often, so clips the data are unsure about still get a turn.",
        "It writes a new settings file (learned.yaml) and never changes config.yaml.",
        "It was tested on simulated outcomes only: it recovers known weights from 500 posts. It has not seen real "
        "posts yet.",
    ]
    st.markdown("\n".join(f"- {x}" for x in lines))
    st.code("uv run promo-finder learn <run> --outcomes posted.csv --out learned.yaml\n"
            "uv run promo-finder learn <run> --outcomes posted.csv --plan 5\n"
            "uv run promo-finder run <next-show> --config learned.yaml", language="bash")
    _try_learn(run)


def _learn_result(run: RunInfo, payload: bytes) -> dict:
    """Run promo_finder.learn in-process on this run with an uploaded outcomes CSV (no model calls, nothing saved).
    Returns the values to show, or {"error": text}."""
    from promo_finder import learn

    with tempfile.TemporaryDirectory(prefix="pf-learn-") as tmp:
        path = Path(tmp) / "outcomes.csv"
        path.write_bytes(payload)
        try:
            pool = learn.load_pool(Path(run.dir))
            obs = learn.read_outcomes(path)
            joined = learn.join(obs, pool)
            res = learn.fit(joined.Z, joined.y, pool.weights, obs.kind, learn.DEFAULT_PRIOR_STRENGTH, pool.Z)
        except learn.LearnError as e:
            return {"error": f"Nothing could be learned from this file: {e}"}
        except Exception as e:  # noqa: BLE001 - a visitor's file can fail in many ways; say so instead of crashing
            return {"error": f"This file could not be read as a results table ({type(e).__name__}: {e})."}
    rows = []
    for t, old, new, sd in zip(learn.TERMS, res.old, res.new, res.sd):
        delta = float(new - old)
        change = fmt.score(0.0) if abs(delta) < 0.0005 else ("+" if delta > 0 else "−") + fmt.score(abs(delta))
        rows.append({"Weight": _TERM_NAMES[t], "Now": fmt.score(old), "Learned": fmt.score(new),
                     "Uncertainty (±1 sd)": "±" + fmt.score(sd), "Change": change})
    notes = list(pool.notes) + list(obs.notes) + list(res.warnings)
    if joined.unknown:
        notes.append(f"{fmt.plural(len(joined.unknown), 'id')} not in this run were skipped.")
    return {"rows": rows, "n": res.n, "metric": obs.metric, "kind": obs.kind, "share": res.data_share, "notes": notes}


def _try_learn(run: RunInfo) -> None:
    with st.expander("Try learn on a results file"):
        st.caption("A .csv with one row per posted promo: its promo_id as in promos.csv (or its rank) and one result "
                   "column, such as hook_rate, hold_rate or approved (1 or 0). It is read on this server, fitted against "
                   "this run's clips and then discarded; nothing is saved and the run's settings do not change.")
        up = st.file_uploader("Results file", type=["csv"], key=f"learn-upload-{run.key}")
        if up is None:
            return
        memo_key = f"_learn_memo_{run.key}"
        memo = st.session_state.get(memo_key)
        if not (isinstance(memo, dict) and memo.get("file_id") == up.file_id):
            with st.spinner("Fitting the weights…"):
                memo = {"file_id": up.file_id, "result": _learn_result(run, up.getvalue())}
            st.session_state[memo_key] = memo
        out = memo["result"]
        if "error" in out:
            st.error(fmt.md_esc(out["error"]), icon=":material/error:")
            return
        df = pd.DataFrame(out["rows"])
        st.dataframe(df, hide_index=True, column_order=list(df.columns), width="content")
        kind = "a rate" if out["kind"] == "continuous" else "approve or reject"
        st.caption(f"Learned from {fmt.plural(out['n'], 'post')} of {fmt.md_esc(out['metric'])} ({kind}). The posts carry "
                   f"{fmt.pct(out['share'])} of the information; the current weights carry the rest.")
        for note in out["notes"]:
            st.caption(f":material/info: {fmt.md_esc(note)}")


def _assumptions(d: dict) -> None:
    st.subheader("Assumptions, and how each was tested", anchor=False)
    rows = [r for r in ((d.get("assumptions") or {}).get("rows") or []) if isinstance(r, dict)]
    if not rows:
        not_measured("assumption checks")
        repo = (data_mod.secret("REPO_URL") or DEFAULT_REPO).rstrip("/")
        st.link_button("Assumptions on GitHub", f"{repo}/blob/main/docs/assumptions.md", icon=":material/open_in_new:")
        return
    false_n = sum(1 for r in rows if re.search(r"\bfalse\b", str(r.get("result") or ""), re.I))
    lead = "Where the brief was silent, the tool assumes these. Each row says how it was tested"
    if false_n:
        lead += f"; {_word(false_n)} turned out false as stated, and the table says so."
    else:
        lead += "."
    st.markdown(lead)
    cols = {"assumption": "Assumption", "evidence": "How it was tested", "result": "Result",
            "if_wrong": "Setting to change if wrong"}
    df = pd.DataFrame([{label: _readable(r.get(k) or "") for k, label in cols.items()} for r in rows])
    st.dataframe(df, hide_index=True, column_order=list(cols.values()), width="stretch")


def _glossary() -> None:
    st.subheader("Glossary", anchor=False)
    df = pd.DataFrame(list(GLOSSARY.items()), columns=["Term", "Meaning"])
    st.dataframe(df, hide_index=True, column_order=["Term", "Meaning"], width="stretch")


def _raw_files(run: RunInfo, d: dict) -> None:
    st.subheader("Raw files", anchor=False)
    have = set(d.get("files") or [])
    shown = 0
    with st.container(horizontal=True, gap="small"):
        for label, rel in RAW_FILES:
            if rel not in have:
                continue
            try:
                payload = (Path(run.dir) / rel).read_bytes()
            except OSError:
                continue
            st.download_button(label, payload, file_name=label, mime=_MIME.get(Path(rel).suffix, "text/plain"),
                               on_click="ignore", icon=":material/download:", key=f"raw-{run.key}-{label}")
            shown += 1
    if shown:
        st.caption("These are the run's own files, unedited.")
    else:
        not_measured("raw files", "This run folder holds none of them.")


# ----------------------------------------------------------------------------------------------------------- view
def render(run: RunInfo | None, data: dict, runs: list[RunInfo]) -> None:
    """Draw the view for the selected run. Never mutate `data`. Never write to st.sidebar."""
    view_header(TITLE, PURPOSE)
    if run is None:
        return
    _diagram()
    cols = st.columns(3, border=True)
    with cols[0]:
        _ai_column(run, data)
    with cols[1]:
        _code_column(run, data)
    with cols[2]:
        _editor_column(run, data)
    st.space("medium")
    _cost(run, data, runs)
    st.space("medium")
    _swap()
    st.space("medium")
    _learning(run, data)
    st.space("medium")
    _assumptions(data)
    st.space("medium")
    _glossary()
    st.space("medium")
    _raw_files(run, data)
