"""Run it view (spec 5.7). Owner: Lane M.

A port of the previous Run tab: the same three modes and CLI flags, the same key isolation (every *_API_KEY, *_AUTH_TOKEN
and *_BASE_URL variable is removed from the child's environment, then only the key this mode may use is added back), the
same 15-minute lock and the same quota and key error detection on the flattened log. New here: four numbered steps, the
2-episode sample, plain captions, stage labels on the progress box and a toast that opens the new run.

Upload scoping (?space=) is cut: runs are tagged "upload" as before and are listed for everyone using the demo, and the
storage caption says so.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from app.ui import data as data_mod
from app.ui import fmt, state
from app.ui.components import view_header
from app.ui.data import RunInfo

TITLE = "Run it on a new show"
PURPOSE = "Upload the next show's scripts and get the same answers: promos, posting order and quality check."

FIXTURES = data_mod.ROOT / "tests" / "fixtures"
SAMPLE_FILES = ("episode-001-synthetic.txt", "episode-002-synthetic.txt")
SAMPLE_FOLDER = "sample-show"
SAMPLE_LINES = 12

LOCK = Path(tempfile.gettempdir()) / "promo_finder_run.lock"
LOCK_S = 15 * 60
POLL_S = 0.5
LOG_TAIL = 40

MODES = ("free", "own", "none")
MODE_LABELS = {"free": "Free open model, hosted here (no key)",
               "own": "Your own key or endpoint, including Kuku's",
               "none": "No AI: keyword rules (instant, free)"}
FREE_KEY_NAMES = ("NVIDIA_API_KEY", "OPENROUTER_API_KEY", "OPENROUTER_API_KEY2", "OPENROUTER_API_KEY3")
NVIDIA_DEMO_MODEL = "nvidia/nvidia/nemotron-3.5-super-vl-120b-a12b"
OPENROUTER_DEMO_MODEL = "openrouter/nvidia/nemotron-3.5-super-vl:free"
OPENROUTER_FREE_CAP = 15          # episodes a free OpenRouter run covers (about 50 requests a day per key)
FREE_JUDGE_TOP_N = 20             # clips the first-time viewer tests in the free mode (--judge-top-n)
OWN_MODEL_DEFAULT = OPENROUTER_DEMO_MODEL

# Log substring -> stage label (spec 5.7). Stages only move forward.
STAGES = (("", "Reading the scripts"),
          ("aliases merged", "Merging speaker names"),
          ("show arc (", "Reading the whole show"),
          ("candidate windows:", "Building clips and running the first-time-viewer test"),
          ("invariants ", "Writing the results"))

QUOTA_MSG = ("The free hosted model's daily quota is used up for today (it resets at 05:30 IST). Nothing was run with it, "
             "and no keyword-rules results were passed off as model results. Use your own key, or keyword rules.")
KEY_MSG = "The provider rejected the key, or no key was given. Check the model name and the key."
OTHER_MSG = "The run stopped with an error; the last lines of its log are below."

STORAGE_CAPTION = ("Your upload and its results stay on this demo server until it restarts, and anyone using this demo "
                   "can see them. Download what you need.")

# Session keys owned by this view.
SRC, SRC_ROOT, SRC_NAME, SRC_KIND = "_run_src", "_run_src_root", "_run_src_name", "_run_src_kind"
UPLOAD_ID, PARSED = "_run_upload_id", "_run_parsed"

_SCRUB = re.compile(r"(?:^|_)(?:API_KEY(?:_?\d+)?|AUTH_TOKEN|BASE_URL)$", re.IGNORECASE)
_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/\S+")


# ----------------------------------------------------------------------------------------------------- pure helpers
def provider_of(model: str) -> str:
    """"openrouter/nvidia/x:free" -> "openrouter"; "none" -> "none"; "" when there is no provider."""
    m = (model or "").strip()
    if m.lower() in ("none", "off", "heuristic"):
        return "none"
    return m.split("/", 1)[0].lower() if _MODEL_RE.fullmatch(m) else ""


def key_var(provider: str) -> str:
    """The environment variable the CLI reads the provider's key from: custom -> PROMO_API_KEY, else <PROVIDER>_API_KEY
    (promo_finder.llm.openai_compat.PROVIDERS follows this rule)."""
    return "PROMO_API_KEY" if provider == "custom" else f"{provider.upper()}_API_KEY"


def build_env(mode: str, server_secrets: dict, pasted_key: str | None, model: str, base_url: str | None) -> dict:
    """The child process environment: os.environ minus every *_API_KEY (and numbered copies), *_AUTH_TOKEN and
    *_BASE_URL variable, plus only the keys this mode may use:
    - "free": the server's key(s) for the free model's own provider (e.g. NVIDIA_API_KEY, or OPENROUTER_API_KEY and its
      numbered rotation keys), taken from `server_secrets`;
    - "own": the visitor's pasted key under the provider's variable, plus PROMO_BASE_URL for custom/<model>;
    - "none" (keyword rules): no key at all.
    The CLI's .env loading is switched off, so a developer's .env can't add keys back."""
    env = {k: v for k, v in os.environ.items() if not _SCRUB.search(k)}
    env["PYTHON_DOTENV_DISABLED"] = "1"
    prov = provider_of(model)
    if mode == "free" and prov not in ("", "none"):
        family = re.compile(re.escape(key_var(prov)) + r"(?:_?\d+)?")
        for name, value in (server_secrets or {}).items():
            if value and family.fullmatch(str(name)):
                env[str(name)] = str(value)
    elif mode == "own" and prov not in ("", "none"):
        if pasted_key and pasted_key.strip() and prov != "ollama":
            env[key_var(prov)] = pasted_key.strip()
        if prov == "custom" and base_url and base_url.strip():
            env["PROMO_BASE_URL"] = base_url.strip()
    return env


def failure_message(log: str) -> tuple[str, str]:
    """(message, icon) for a run that did not finish, from its log. The CLI wraps long lines, so match on the
    flattened text."""
    joined = " ".join((log or "").split())
    if "daily quota" in joined:
        return QUOTA_MSG, ":material/hourglass_empty:"
    if "HTTP 401" in joined or "HTTP 403" in joined or "no API key" in joined:
        return KEY_MSG, ":material/key_off:"
    return OTHER_MSG, ":material/error:"


def stage_of(line: str) -> int:
    """Index into STAGES of the last stage marker in a log line (0 when none)."""
    best = 0
    for i, (marker, _label) in enumerate(STAGES):
        if marker and marker in line:
            best = i
    return best


def status_label(stage: int, elapsed_s: float) -> str:
    return f"{STAGES[stage][1]} · {fmt.num(int(elapsed_s))} s"


def sample_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in SAMPLE_FILES:
            z.write(FIXTURES / name, f"{SAMPLE_FOLDER}/{name}")
    return buf.getvalue()


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")[:40] or "show"


def _one_decimal(x: float) -> str:
    """70.5 -> "70.5" (fmt.secs without the unit)."""
    return fmt.secs(x)[:-2]


# ----------------------------------------------------------------------------------------------------- source folder
def _drop_source() -> None:
    ss = st.session_state
    root = ss.pop(SRC_ROOT, None)
    for k in (SRC, SRC_NAME, PARSED):
        ss.pop(k, None)
    if ss.pop(SRC_KIND, None) == "upload":
        ss.pop(UPLOAD_ID, None)  # the same upload can be unpacked and run again
    if root and Path(root).name.startswith("pf-") and Path(root).parent == Path(tempfile.gettempdir()):
        shutil.rmtree(root, ignore_errors=True)


def _set_source(folder: Path, root: Path, name: str, kind: str) -> None:
    ss = st.session_state
    old = ss.get(SRC_ROOT)
    if old and old != str(root) and Path(old).name.startswith("pf-"):
        shutil.rmtree(old, ignore_errors=True)
    ss[SRC], ss[SRC_ROOT], ss[SRC_NAME], ss[SRC_KIND] = str(folder), str(root), name, kind
    ss.pop(PARSED, None)


def _use_sample() -> None:
    """on_click: copy the two synthetic fixtures into a fresh temp folder named like a show."""
    root = Path(tempfile.mkdtemp(prefix="pf-"))
    dst = root / SAMPLE_FOLDER
    dst.mkdir()
    for name in SAMPLE_FILES:
        shutil.copy(FIXTURES / name, dst / name)
    _set_source(dst, root, "the 2-episode sample", "sample")


def _unpack(up) -> tuple[Path, Path]:
    root = Path(tempfile.mkdtemp(prefix="pf-"))
    dst = root / _slug(Path(up.name).stem)
    dst.mkdir()
    if up.name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(up.getvalue())) as z:
            z.extractall(dst)  # zipfile drops absolute paths and ".." parts
    else:
        (dst / Path(up.name).name).write_bytes(up.getvalue())
    return dst, root


# ----------------------------------------------------------------------------------------------------- steps
def _step(n: int, title: str):
    box = st.container(border=True)
    box.markdown(f"**{fmt.num(n)} · {title}**")
    return box


def _scripts() -> None:
    ss = st.session_state
    with _step(1, "The scripts"):
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        up = c1.file_uploader("Screenplay pack", type=["zip", "txt", "md", "fountain"], key="run_upload",
                              help="A .zip with one file per episode, or one file holding every episode (.txt, .md or "
                                   ".fountain). 50 MB max.")
        c2.button("Try the 2-episode sample", icon=":material/science:", on_click=_use_sample, key="run_sample",
                  help="A made-up 2-episode script written for tests, not Kuku's.")
        if up is None and ss.get(SRC_KIND) == "upload":
            _drop_source()  # the visitor removed the file
        if up is not None and ss.get(UPLOAD_ID) != up.file_id:
            ss[UPLOAD_ID] = up.file_id
            try:
                dst, root = _unpack(up)
            except (zipfile.BadZipFile, OSError, ValueError) as e:
                st.error(f"This file could not be opened: {fmt.md_esc(e)}.", icon=":material/error:")
            else:
                _set_source(dst, root, up.name, "upload")
        if ss.get(SRC) and Path(ss[SRC]).exists():
            st.caption(f"Using {fmt.md_esc(ss.get(SRC_NAME) or 'your upload')}.")
        with st.expander("What the scripts should look like"):
            try:
                head = (FIXTURES / SAMPLE_FILES[0]).read_text(encoding="utf-8").splitlines()[:SAMPLE_LINES]
            except OSError:
                head = []
            if head:
                st.code("\n".join(head), language=None)
            st.caption("One text file per episode, or one file with every episode. Every block needs its time, like "
                       "Start: (00:01.00) | End: (00:02.50).")
            try:
                st.download_button("Download the sample (.zip)", sample_zip(), file_name="sample-show.zip",
                                   mime="application/zip", on_click="ignore", icon=":material/download:",
                                   key="run_sample_zip")
            except OSError:
                pass


def _parse(src: Path) -> dict:
    """Parse once per source folder; kept in session state."""
    ss = st.session_state
    cached = ss.get(PARSED)
    if isinstance(cached, dict) and cached.get("src") == str(src):
        return cached
    from promo_finder.parse import parse_report, parse_show

    skipped: list = []
    try:
        eps = parse_show(src, skipped)
        rep = parse_report(eps) if eps else {}
        res = {"src": str(src), "ok": bool(eps), "rep": rep, "skipped": len(skipped),
               "error": "" if eps else "no episodes were found"}
    except Exception as e:  # noqa: BLE001 - any parse failure is shown to the visitor
        first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        res = {"src": str(src), "ok": False, "rep": {}, "skipped": len(skipped), "error": first.rstrip(".")}
    ss[PARSED] = res
    return res


def _check() -> bool:
    ss = st.session_state
    with _step(2, "Input check"):
        src = ss.get(SRC)
        if not src or not Path(src).exists():
            st.caption("Upload scripts or try the sample, and they are checked here in a second.")
            return False
        res = _parse(Path(src))
        if not res["ok"]:
            st.error(f"This doesn't look like a timestamped screenplay: {fmt.md_esc(res['error'])}. Compare it with the "
                     "sample above.", icon=":material/error:")
            return False
        rep = res["rep"]
        msg = (f"Read {fmt.plural(rep.get('episodes', 0), 'episode')}: {fmt.num(rep.get('total_blocks', 0))} timed blocks, "
               f"{_one_decimal(rep.get('duration_min_s', 0))}–{fmt.secs(rep.get('duration_max_s', 0))} each.")
        k = res["skipped"]
        if k:
            msg += f" Skipped {fmt.num(k)} " + ("file that isn't a screenplay." if k == 1 else "files that aren't screenplays.")
        w = int(rep.get("episodes_with_warnings") or 0)
        if w:
            msg += f" {fmt.plural(w, 'episode')} had small format problems, all handled."
        st.success(msg, icon=":material/task_alt:")
        with st.expander("Details"):
            rows = [{"Episode": fmt.num(r.get("episode", 0)),
                     "Length (s)": _one_decimal(r.get("duration_s", 0)),
                     "Scenes": fmt.num(r.get("scenes", 0)),
                     "Lines": fmt.num(r.get("dialogue_lines", 0)),
                     "Parsed": fmt.pct(r.get("parse_rate", 0)),
                     "Notes": "; ".join(str(x) for x in (r.get("warnings") or []))}
                    for r in rep.get("per_episode") or [] if isinstance(r, dict)]
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, hide_index=True, column_order=list(df.columns), width="stretch")
        return True


def _server_secrets(demo_model: str) -> dict:
    names = list(FREE_KEY_NAMES)
    prov = provider_of(demo_model)
    if prov not in ("", "none"):
        base = key_var(prov)
        names += [base] + [f"{base}{s}" for s in ("2", "_2", "3", "_3")]
    out = {}
    for n in dict.fromkeys(names):
        v = data_mod.secret(n)
        if v:
            out[n] = v
    return out


def _free_setup(runs: list[RunInfo]) -> dict:
    nvidia = bool(data_mod.secret("NVIDIA_API_KEY"))
    demo_model = data_mod.secret("PROMO_DEMO_MODEL") or (NVIDIA_DEMO_MODEL if nvidia else OPENROUTER_DEMO_MODEL)
    prov = provider_of(demo_model)
    try:
        cap = int(data_mod.secret("DEMO_MAX_EPISODES") or (0 if prov == "nvidia" else OPENROUTER_FREE_CAP))
    except ValueError:
        cap = 0
    secrets = _server_secrets(demo_model)
    family = re.compile(re.escape(key_var(prov)) + r"(?:_?\d+)?") if prov not in ("", "none") else None
    ok = bool(family and any(family.fullmatch(k) for k in secrets))
    if prov == "nvidia":
        extra = ["--reasoning-off", "--rpm", "35", "--concurrency", "4", "--no-entailment",
                 "--judge-top-n", str(FREE_JUDGE_TOP_N)]
    else:
        extra = ["--batch", "5", "--reasoning-off", "--rpm", "12", "--concurrency", "2", "--no-entailment",
                 "--judge-top-n", str(FREE_JUDGE_TOP_N)]
    name = ("NVIDIA " if "nemotron" in demo_model.lower() else "") + fmt.model_short(demo_model)
    official = next((r for r in runs if r.role == "official"), None)
    same = bool(official and fmt.model_short(official.model) == fmt.model_short(demo_model))
    lead = name + (", the model behind the official results," if same else "")
    if not ok:
        caption = "This deployment has no free model key. Use your own key, or keyword rules."
    elif prov == "nvidia":
        shortlist = _config_shortlist()
        caption = (f"{lead} through NVIDIA's free API (about 40 requests a minute, shared by everyone using this app). "
                   "To fit that limit it skips the continuity check and tests the top "
                   f"{fmt.num(FREE_JUDGE_TOP_N)} clips" + (f", not {fmt.num(shortlist)}," if shortlist else "")
                   + " on the first-time viewer.")
    elif prov == "openrouter":
        caption = (f"{lead} through OpenRouter's free tier: about 50 requests a day per key, so a free run covers "
                   + (f"the first {fmt.plural(cap, 'episode')}." if cap else "only part of a long show."))
    else:
        caption = f"{lead} through this deployment's own free key."
    return {"model": demo_model, "cap": cap, "secrets": secrets, "ok": ok, "extra": extra, "caption": caption}


def _config_shortlist() -> int:
    try:
        import yaml

        cfg = yaml.safe_load((data_mod.ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
        return int(((cfg.get("judge") or {}).get("top_n")) or 0)
    except Exception:  # noqa: BLE001
        return 0


def _model(runs: list[RunInfo]) -> dict:
    """Step 3. Returns mode, model, flags, env inputs and the reason the run can't start (or "")."""
    free = _free_setup(runs)
    with _step(3, "Model"):
        captions = [free["caption"],
                    "Any listed provider, or any OpenAI-compatible endpoint: choose custom/<model id> and paste its base "
                    "URL. The key is used for this run only and never saved. The model's answers are cached on this server "
                    "until it restarts.",
                    "Scores lines with a word list (slaps, threats, a question at the end). Use it to check that the format "
                    "reads, or as the baseline the AI is compared with. There is no first-time viewer and no promise check."]
        mode = st.radio("Model", MODES, format_func=MODE_LABELS.get, captions=[fmt.md_esc(c) for c in captions],
                        index=0 if free["ok"] else 2, key="run_mode")
        out = {"mode": mode, "model": "none", "extra": [], "cap": 0, "secrets": {}, "key": None, "base": None,
               "blocked": ""}
        if mode == "free":
            st.warning("Free tiers can log what you send them. Don't upload confidential scripts in the free mode; use "
                       "your own key or endpoint.", icon=":material/privacy_tip:")
            if free["ok"]:
                out.update(model=free["model"], extra=free["extra"], cap=free["cap"], secrets=free["secrets"])
            else:
                out["blocked"] = "This deployment has no free model key. Choose your own key, or keyword rules."
        elif mode == "own":
            c1, c2 = st.columns([2, 1])
            model = c1.text_input("Model (provider/model)", value=OWN_MODEL_DEFAULT, key="run_own_model",
                                  help="nvidia/… (free), openrouter/…, gemini/…, groq/…, mistral/…, openai/…, ollama/…, or "
                                       "custom/<model id> for any OpenAI-compatible endpoint.").strip()
            key = c2.text_input("API key", type="password", key="run_own_key", help="Used for this run only, never saved.")
            prov = provider_of(model)
            base = ""
            if prov == "custom":
                base = st.text_input("Base URL", key="run_own_base",
                                     placeholder="https://your-gateway.example.com/v1").strip()
            c3, c4 = st.columns(2)
            fast = c3.toggle("Send 8 requests at a time (only for paid or uncapped keys)",
                             value=not model.endswith(":free"), key="run_own_fast")
            votes = c4.selectbox("Read each episode", [1, 2, 3], key="run_own_votes",
                                 format_func=lambda n: f"{n} time{'s' if n > 1 else ''}",
                                 help="2 readings raised simulated top-10 agreement from 40% to 52%, for twice the "
                                      "requests.")
            out.update(model=model or "none", key=key, base=base,
                       extra=(["--fast"] if fast else ["--batch", "5", "--reasoning-off", "--rpm", "12"])
                       + ["--votes", str(int(votes))])
            if not prov:
                out["blocked"] = "Enter the model as provider/model, for example openrouter/<model id>."
            elif prov == "custom" and not re.match(r"https?://\S+$", base):
                out["blocked"] = "Paste the endpoint's base URL (https://…) for a custom model."
            elif not (key or "").strip() and prov not in ("ollama", "none"):
                st.caption(":material/key: Paste a key for that provider to run with it.")
        return out


def _episodes(cap: int, mode: str) -> int:
    with _step(4, "Episodes"):
        all_eps = st.toggle("All episodes", value=(cap == 0), key=f"run_all_{mode}")
        if all_eps:
            return 0
        return int(st.number_input("First N episodes", min_value=1, max_value=500, value=cap or 10, step=1,
                                   key=f"run_limit_{mode}"))


def _lock_age() -> float | None:
    try:
        return time.time() - LOCK.stat().st_mtime
    except OSError:
        return None


def _take_lock() -> bool:
    """Create the lock atomically; a lock older than 15 minutes is stale and replaced."""
    age = _lock_age()
    if age is not None and age >= LOCK_S:
        LOCK.unlink(missing_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    return True


def _newest_upload(since: float) -> Path | None:
    found = [p.parent for p in data_mod.OUT.glob("*/*-upload/promos.json") if p.stat().st_mtime >= since - 1]
    return max(found, key=lambda p: (p / "promos.json").stat().st_mtime) if found else None


def _run(src: Path, cfg: dict, limit: int) -> None:
    """Start the CLI, stream its log into a status box with stage labels, then open the new run or explain the
    failure. The log goes to a file, so a visitor who leaves the tab never blocks the child on a full pipe."""
    if not _take_lock():
        st.error("Another run started a moment ago. Try again when it has finished.", icon=":material/hourglass_empty:")
        return
    env = build_env(cfg["mode"], cfg["secrets"], cfg["key"], cfg["model"], cfg["base"])
    env["COLUMNS"] = "220"  # keep the CLI's messages on one line each
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, "-m", "promo_finder.cli", "run", str(src), "--model", cfg["model"],
           "--judge-model", cfg["model"], "--out", str(data_mod.OUT), "--tag", "upload"] + cfg["extra"]
    if limit:
        cmd += ["--limit", str(int(limit))]
    st.caption("Stay on this tab until it finishes. If you switch tabs, the progress display stops, but the run keeps "
               "going and appears under Results when it is done.")
    fd, log_name = tempfile.mkstemp(prefix="pf-log-", suffix=".txt")
    os.close(fd)
    log_path = Path(log_name)
    lines: list[str] = []
    proc = None
    started = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as sink:
            proc = subprocess.Popen(cmd, stdout=sink, stderr=subprocess.STDOUT, env=env, cwd=str(data_mod.ROOT))
        stage = 0
        with st.status(status_label(0, 0), expanded=False) as box:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                pending = ""
                while True:
                    finished = proc.poll() is not None
                    chunk = fh.read()
                    if chunk:
                        pending += chunk
                        *complete, pending = pending.split("\n")
                        if finished and pending:
                            complete, pending = complete + [pending], ""
                        for line in complete:
                            line = line.rstrip()
                            if line.strip():
                                lines.append(line)
                                stage = max(stage, stage_of(line))
                                st.text(line[:200])
                    box.update(label=status_label(stage, time.time() - started))
                    if finished:
                        break
                    time.sleep(POLL_S)
            ok = proc.returncode == 0
            box.update(label=("Done" if ok else "Stopped with an error") + f" · {fmt.num(int(time.time() - started))} s",
                       state="complete" if ok else "error")
    finally:
        if proc is None or proc.poll() is not None:  # an interrupted display leaves a running child its lock and files
            LOCK.unlink(missing_ok=True)
            _drop_source()
            log_path.unlink(missing_ok=True)
    new = _newest_upload(started) if proc.returncode == 0 else None
    if new is not None:
        try:
            promos = json.loads((new / "promos.json").read_text(encoding="utf-8"))
            metrics = json.loads((new / "eval" / "metrics.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            promos, metrics = [], {}
        n = len(promos) if isinstance(promos, list) else 0
        eps = metrics.get("episodes") if isinstance(metrics, dict) else None
        eps = eps if isinstance(eps, int) else 0
        secs = time.time() - started  # what the visitor waited, start-up included
        st.toast(f"Done: {fmt.plural(n, 'promo')} from {fmt.plural(eps, 'episode')} in {fmt.secs(secs)}. "
                 "Showing them now.", icon=":material/task_alt:", duration="long")
        st.cache_data.clear()
        run_key = next((r.key for r in data_mod.list_runs() if Path(r.dir).resolve() == new.resolve()), None)
        state.goto_later("Promos", run_key)
        return
    msg, icon = failure_message("\n".join(lines))
    st.error(msg, icon=icon)
    if msg == OTHER_MSG:
        st.code("\n".join(lines[-LOG_TAIL:]) or "The run printed nothing.", language=None)


# ----------------------------------------------------------------------------------------------------- view
def render(run: RunInfo | None, data: dict, runs: list[RunInfo]) -> None:
    """Draw the view for the selected run. Never mutate `data`. Never write to st.sidebar. Works with run None."""
    view_header(TITLE, PURPOSE)
    _scripts()
    parsed = _check()
    cfg = _model(runs or [])
    limit = _episodes(cfg["cap"], cfg["mode"])

    age = _lock_age()
    busy = age is not None and age < LOCK_S
    if not parsed:
        why = "Upload scripts or try the sample first."
    elif busy:
        why = (f"Another run is in progress; the next one can start in about "
               f"{fmt.num(max(1, math.ceil((LOCK_S - age) / 60)))} min.")
    else:
        why = cfg["blocked"]
    clicked = st.button("Find promos", type="primary", icon=":material/play_arrow:", key="run_go",
                        disabled=bool(why), help=why or "Runs the whole pipeline on these scripts with the chosen model.")
    status_area = st.container()
    st.caption(STORAGE_CAPTION)
    with st.expander("Run it on your own machine"):
        st.code("uv sync\nuv run promo-finder check --model <provider/model>\n"
                "uv run promo-finder run <folder> --model <provider/model> --fast", language="bash")
    if clicked and not why:
        with status_area:
            _run(Path(st.session_state[SRC]), cfg, limit)
