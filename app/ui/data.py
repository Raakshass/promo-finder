"""Where results come from (spec 4.3 and 6): the outputs fetch, run discovery and labels, and the cached run loader.

OUT is PROMO_OUTPUTS_DIR when set, else <repo>/outputs. Tests point it at <repo>/submission/outputs.
Upload scoping by ?space= is cut from P0: every run started in Run it (run id ending in "-upload" or "-upload-<x>")
is listed for everyone, as today.
"""
from __future__ import annotations

import io
import json
import os
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import streamlit as st
import yaml

from app.ui import fmt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("PROMO_OUTPUTS_DIR") or ROOT / "outputs")

OFFICIAL_MARK = "OFFICIAL"
_UPLOAD = re.compile(r"-upload(?:-[0-9a-z]+)?$")

# Evidence runs, in display order: folder prefix -> (key, label, caption, short). Spec 6.1.
_EVIDENCE = {
    "previous_version": ("first-version", "Evidence · First version",
                         "The first version's top 20, picked by score from anywhere in the show, with no editor's checks and "
                         "no opening-first order. The blind A/B compares the official slate with it.", "First version"),
    "heuristic_only": ("keyword-rules", "Evidence · No AI: keyword rules",
                       "The same pipeline with a word list instead of the model: instant and free. The baseline the AI must "
                       "beat.", "Keyword rules"),
    "cue_blind_ablation": ("cues-hidden", "Evidence · Writers' cues hidden",
                           "The model never sees the writers' 'cliffhanger' and 'cut to black' notes.", "Cues hidden"),
    "stability_base": ("consistency-a", "Evidence · Consistency run A",
                       "A fresh run of the same model (earlier version, every answer new). With run B it measures how much "
                       "line scores change between runs.", "Consistency A"),
    "stability_rerun": ("consistency-b", "Evidence · Consistency run B",
                        "The same run repeated a minute later with the answer cache off.", "Consistency B"),
}
_UPLOAD_CAPTION = "Started in Run it. Kept until this demo server restarts; anyone using this demo can see it."
_LOCAL_CAPTION = "Command-line run on this machine. Not part of the submission."


def secret(name: str) -> str | None:
    """os.environ first, then st.secrets; None when unset or when there is no secrets file."""
    v = os.environ.get(name)
    if v:
        return v
    try:
        if name in st.secrets:
            v = st.secrets[name]
            return str(v) if v not in (None, "") else None
    except Exception:  # noqa: BLE001 - no secrets file, or a malformed one
        return None
    return None


def fetch_outputs() -> str | None:
    """Pull outputs.zip from OUTPUTS_URL into OUT/_fetched, once per container (skipped when
    OUT/_fetched/*/*/promos.json exists). Drive share links are rewritten to direct downloads.
    Returns None on success or when there is nothing to do, else the error text. The caller shows the spinner,
    the friendly error and the 60 s back-off."""
    url = secret("OUTPUTS_URL")
    if not url or any((OUT / "_fetched").glob("*/*/promos.json")):
        return None
    m = re.search(r"drive\.google\.com/file/d/([^/]+)", url)
    if m:
        url = f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    if not url.lower().startswith(("https://", "http://")):
        return "OUTPUTS_URL must start with https:// or http://"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "promo-finder"})
        payload = urllib.request.urlopen(req, timeout=120).read()  # noqa: S310 - scheme checked above
        (OUT / "_fetched").mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            z.extractall(OUT / "_fetched")
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    _list_runs.clear()
    return None


class RunReadError(Exception):
    """A run file exists but cannot be read. `file` is its path relative to the run folder."""

    def __init__(self, file: str, cause: Exception):
        super().__init__(f"{file}: {type(cause).__name__}: {cause}")
        self.file = file
        self.cause = cause


@dataclass(frozen=True)
class RunInfo:
    key: str            # URL-safe, stable (spec 6.1)
    dir: Path
    role: Literal["official", "evidence", "upload", "local"]
    label: str          # radio label (spec 6.1)
    caption: str        # one-sentence purpose (spec 6.1)
    short: str          # popover label: "Official", "First version", "Your run 21:48"...
    show_title: str     # fmt.show_title(parent folder name)
    run_id: str         # metrics.run_id, else the folder name after any "<label>__" prefix
    model: str          # metrics.model.model (or run_log.args.model); "none" for keyword runs
    episodes: int       # metrics.episodes, else parse_report.episodes, else 0
    order: int


def stamp(run_dir: Path | str) -> float:
    """mtime of the run's promos.json (the load_run cache key); 0.0 when missing."""
    try:
        return (Path(run_dir) / "promos.json").stat().st_mtime
    except OSError:
        return 0.0


def _json(path: Path) -> dict:
    try:
        v = json.loads(path.read_text(encoding="utf-8"))
        return v if isinstance(v, dict) else {}
    except (OSError, ValueError):
        return {}


def _slug(s: str) -> str:
    """Same as the CLI's run-folder slug."""
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def _tag(run_id: str, model_arg: str) -> str:
    """The run id after the timestamp and the model slug: "…--final-v4" -> "final-v4"."""
    rest = re.sub(r"^\d{8}-\d{6}-", "", run_id)
    m = (model_arg or "none").strip()
    if m.lower() in ("", "none", "off", "heuristic") or "/" not in m:
        slug = "heuristic"
    else:
        provider, model = m.split("/", 1)
        slug = _slug(f"{provider.lower()}-{model}")
    if rest.startswith(slug):
        rest = rest[len(slug):]
    return rest.strip("-")


def _describe(d: Path) -> dict:
    """Facts for one run folder, read from its own files."""
    metrics = _json(d / "eval" / "metrics.json")
    run_log = _json(d / "run_log.json")
    args = run_log.get("args") or {}
    prefix, sep, rest = d.name.partition("__")
    folder_run_id = rest if sep else d.name
    run_id = str(metrics.get("run_id") or folder_run_id)
    mm = metrics.get("model") or {}
    model = str(mm.get("model") or args.get("model") or "none")
    if str(mm.get("provider") or "").lower() == "none" or model.lower() in ("", "none", "off", "heuristic"):
        model = "none"
    episodes = metrics.get("episodes")
    if not isinstance(episodes, int):
        episodes = _json(d / "parse_report.json").get("episodes")
    show_slug = d.parent.name
    if show_slug.startswith("_") and args.get("show_dir"):
        show_slug = Path(str(args["show_dir"])).name
    return {
        "dir": d, "prefix": prefix if sep else "", "run_id": run_id, "model": model,
        "model_arg": str(args.get("model") or ""), "episodes": int(episodes) if isinstance(episodes, int) else 0,
        "limit": int(args.get("limit") or 0), "official": (d / OFFICIAL_MARK).exists(),
        "recall": ((metrics.get("cliff_localisation") or {}).get("recall_all")),
        "show_title": fmt.show_title(show_slug),
    }


def _hhmmss(run_id: str) -> str:
    m = re.match(r"\d{8}-(\d{6})", run_id)
    return m.group(1) if m else "000000"


def _hhmm(run_id: str) -> str:
    s = fmt.run_time_short(run_id)
    return s.rsplit(" ", 1)[-1] if s else ""


@st.cache_data(ttl=10, show_spinner=False)
def _list_runs(out: str, space: str | None) -> list[RunInfo]:
    root = Path(out)
    dirs = {p.parent.resolve() for p in root.glob("*/*/promos.json")}
    dirs |= {p.parent.resolve() for p in root.glob("_fetched/**/promos.json")}
    facts = [_describe(d) for d in sorted(dirs)]

    def kind(f: dict) -> str:
        if f["official"]:
            return "official"
        if f["prefix"] in _EVIDENCE:
            return "evidence"
        if _UPLOAD.search(f["run_id"]):
            return "upload"
        return "local"

    # De-duplicate by run_id: official copy first, then evidence, upload, local.
    rank = {"official": 0, "evidence": 1, "upload": 2, "local": 3}
    best: dict[str, dict] = {}
    for f in facts:
        f["kind"] = kind(f)
        cur = best.get(f["run_id"])
        if cur is None or rank[f["kind"]] < rank[cur["kind"]]:
            best[f["run_id"]] = f
    runs = list(best.values())

    officials = sorted((f for f in runs if f["kind"] == "official"), key=lambda f: f["run_id"], reverse=True)
    for extra in officials[1:]:  # more than one OFFICIAL marker: only the newest is "the" official run
        extra["kind"] = "local"
    official = officials[0] if officials else None
    official_recall = official["recall"] if official else None

    out_runs: list[dict] = []
    if official:
        cap = (f"{official['episodes']} episodes · {fmt.model_short(official['model'])}"
               + (" (free tier)" if fmt.is_free(official["model"]) else "")
               + (f" · run {fmt.run_time(official['run_id'])}" if fmt.run_time(official["run_id"]) else "")
               + ". The results the solution doc reports.")
        out_runs.append(dict(official, key="official", role="official", label="Official results", caption=cap,
                             short="Official"))

    used: set[str] = {"official"}

    def unique(key: str) -> str:
        k, i = key, 2
        while k in used:
            k, i = f"{key}-{i}", i + 1
        used.add(k)
        return k

    for f in sorted((f for f in runs if f["kind"] == "upload"), key=lambda f: f["run_id"], reverse=True):
        label = (f"Your run · {f['show_title']} · {f['episodes']} episodes · {fmt.model_short(f['model'])}"
                 + (f" · {fmt.run_time_short(f['run_id'])}" if fmt.run_time_short(f["run_id"]) else "")
                 + (f" · first {f['limit']} episodes" if f["limit"] > 0 else ""))
        out_runs.append(dict(f, key=unique(f"upload-{_hhmmss(f['run_id'])}"), role="upload", label=label,
                             caption=_UPLOAD_CAPTION, short=f"Your run {_hhmm(f['run_id'])}".strip()))

    evid = [f for f in runs if f["kind"] == "evidence"]
    order = list(_EVIDENCE)
    for f in sorted(evid, key=lambda f: (order.index(f["prefix"]), f["run_id"])):
        key, label, caption, short = _EVIDENCE[f["prefix"]]
        if f["prefix"] == "previous_version" and fmt.run_date(f["run_id"]):
            label = f"{label} ({fmt.run_date(f['run_id'])})"
        if f["prefix"] == "cue_blind_ablation" and f["recall"] is not None and official_recall is not None:
            caption = (f"The model never sees the writers' 'cliffhanger' and 'cut to black' notes, and still finds their "
                       f"cliffhanger in {fmt.pct(f['recall'])} of episodes ({fmt.pct(official_recall)} with them).")
        out_runs.append(dict(f, key=unique(key), role="evidence", label=label, caption=caption, short=short))

    for f in sorted((f for f in runs if f["kind"] == "local"), key=lambda f: f["run_id"], reverse=True):
        tag = _tag(f["run_id"], f["model_arg"])
        label = " · ".join(x for x in (f["show_title"], fmt.model_short(f["model"]), fmt.run_time_short(f["run_id"]), tag) if x)
        out_runs.append(dict(f, key=unique(f"local-{f['run_id']}"), role="local", label=label, caption=_LOCAL_CAPTION,
                             short=f"Local {_hhmm(f['run_id'])}".strip()))

    return [RunInfo(key=r["key"], dir=r["dir"], role=r["role"], label=r["label"], caption=r["caption"], short=r["short"],
                    show_title=r["show_title"], run_id=r["run_id"], model=r["model"], episodes=r["episodes"], order=i)
            for i, r in enumerate(out_runs)]


def list_runs(space: str | None = None) -> list[RunInfo]:
    """Folders with promos.json under OUT/*/* and OUT/_fetched/**, de-duplicated by run_id (the copy holding OFFICIAL
    wins). Order: official, your runs (newest first), evidence (spec 6.1 order), local (newest first).
    `space` is accepted for the frozen signature but ignored: upload scoping is cut from P0. Cached for 10 s per OUT;
    `list_runs.clear()` or `st.cache_data.clear()` drops the cache."""
    return _list_runs(str(OUT), None)


list_runs.clear = _list_runs.clear  # type: ignore[attr-defined]


def _read(d: Path, rel: str, kind: str, files_read: list[str]):
    p = d / rel
    empty = {"json": {}, "yaml": {}, "text": ""}[kind]
    if not p.exists():
        return empty
    try:
        raw = p.read_text(encoding="utf-8")
        if kind == "json":
            v = json.loads(raw)
        elif kind == "yaml":
            v = yaml.safe_load(raw)
        else:
            v = raw
    except Exception as e:  # noqa: BLE001
        raise RunReadError(rel, e) from e
    files_read.append(rel)
    if kind in ("json", "yaml") and v is None:
        return empty
    return v


@st.cache_data(show_spinner=False, max_entries=8)
def load_run(run_dir: str, stamp: float) -> dict:
    """stamp = mtime of promos.json (see stamp()). Missing files give empty values ({} / [] / "") and never raise.
    A file that exists but cannot be parsed raises RunReadError naming it."""
    d = Path(run_dir)
    seen: list[str] = []
    promos = _read(d, "promos.json", "json", seen)
    out = {
        "promos": promos if isinstance(promos, list) else [],
        "metrics": _read(d, "eval/metrics.json", "json", seen),
        "cfg": _read(d, "config.yaml", "yaml", seen),
        "run_log": _read(d, "run_log.json", "json", seen),
        "arc": _read(d, "show_arc.json", "json", seen),
        "beats": {},
        "parse": _read(d, "parse_report.json", "json", seen),
        "aliases": _read(d, "aliases.yaml", "yaml", seen),
        "promoability": _read(d, "eval/promoability.json", "json", seen),
        "stability": _read(d, "eval/stability.json", "json", seen),
        "votes": _read(d, "eval/votes_simulation.json", "json", seen),
        "ab": _read(d, "eval/ab_vs_previous.json", "json", seen),
        "benchmark": _read(d, "eval/external_benchmark_slate.json", "json", seen),
        "dramaad": _read(d, "eval/dramaad_stats.json", "json", seen),
        "assumptions": _read(d, "eval/assumption_checks.json", "json", seen),
        "check_exp": _read(d, "eval/check_experiment.json", "json", seen),
        "check_exp_ends_open": _read(d, "eval/check_experiment_ends_open.json", "json", seen),
        "promos_csv": _read(d, "promos.csv", "text", seen),
        "promos_md": _read(d, "promos.md", "text", seen),
        "per_episode_csv": _read(d, "promos_by_episode.csv", "text", seen),
        "cut_script": _read(d, "cut_promos.sh", "text", seen),
        "failures_md": _read(d, "eval/failures.md", "text", seen),
        "official": (d / OFFICIAL_MARK).exists(),
    }
    for k in ("metrics", "cfg", "run_log", "arc", "parse", "aliases", "promoability", "stability", "votes", "ab",
              "benchmark", "dramaad", "assumptions", "check_exp", "check_exp_ends_open"):
        if not isinstance(out[k], dict):
            out[k] = {}
    beats: dict[int, dict] = {}
    for f in sorted((d / "beats").glob("ep-*.json")):
        rel = f"beats/{f.name}"
        b = _read(d, rel, "json", seen)
        if isinstance(b, dict):
            m = re.search(r"(\d+)", f.stem)
            ep = b.get("episode") if isinstance(b.get("episode"), int) else (int(m.group(1)) if m else None)
            if ep is not None:
                beats[ep] = b
    out["beats"] = beats
    if beats:
        out["episodes_list"] = sorted(beats)
    else:
        out["episodes_list"] = sorted(int(r["episode"]) for r in out["parse"].get("per_episode") or []
                                      if isinstance(r, dict) and isinstance(r.get("episode"), int))
    out["files"] = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()) if d.exists() else []
    return out
