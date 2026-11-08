"""Parts of the show, the posting-order replay, "Why #k" sentences and the rule box (spec 5.4). Owner: Lane R.

The replay mirrors promo_finder.rank.diversity_order inside each part (promo_finder.slate.build), restricted to the
promos on the slate. Every share, penalty and cap is read from the run's config.yaml; none is typed here.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.ui import fmt
from app.ui.glossary import PART_NAMES, TYPE_WORDS

try:  # pure function; the app may import it (spec 1)
    from promo_finder.slate import BLOCKS, block_of
except Exception:  # noqa: BLE001
    BLOCKS = ("opening", "middle", "late")
    block_of = None

_VARIETY = ("same_episode_penalty", "same_type_penalty", "max_per_episode_top10")
_NTH = {2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}
_TIMES = {1: "", 2: " twice"}


@dataclass(frozen=True)
class Part:
    key: str
    name: str
    first_ep: int
    last_ep: int
    n: int
    first_rank: int
    last_rank: int


@dataclass(frozen=True)
class Rival:
    rank: int
    score: float
    counted: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Placement:
    rank: int
    part: str
    score: float
    counted: float
    reasons: tuple[str, ...]
    rivals: tuple[Rival, ...]
    first_in_part: bool


def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def _ep(p: dict) -> int | None:
    e = _d(p.get("flags")).get("episode")
    if isinstance(e, int):
        return e
    segs = p.get("segments") or []
    return segs[0].get("episode") if segs and isinstance(segs[0].get("episode"), int) else None


def _ranked(promos: list[dict]) -> list[dict]:
    return sorted((p for p in promos or [] if isinstance(p.get("rank"), int)), key=lambda p: p["rank"])


def _block(p: dict, cfg: dict, lo: int, hi: int) -> str | None:
    b = _d(p.get("flags")).get("block")
    if b in BLOCKS:
        return b
    e = _ep(p)
    if block_of is None or e is None or not _d(cfg).get("slate"):
        return None
    try:
        return block_of(e, lo, hi, cfg)
    except (KeyError, TypeError, ValueError):
        return None


def parts(promos: list[dict], cfg: dict, episodes: list[int]) -> list[Part]:
    """Only parts with promos, in posting order. Episode ranges come from promo_finder.slate.block_of over the run's
    episodes; without a slate config they fall back to the episodes the part's promos come from."""
    ranked = _ranked(promos)
    eps = sorted({int(e) for e in episodes or []}) or sorted({e for p in ranked if (e := _ep(p)) is not None})
    if not ranked or not eps:
        return []
    lo, hi = eps[0], eps[-1]
    out: list[Part] = []
    for key in BLOCKS:
        mine = [p for p in ranked if _block(p, cfg, lo, hi) == key]
        if not mine:
            continue
        span: list[int] = []
        if block_of is not None and _d(cfg).get("slate"):
            try:
                span = [e for e in eps if block_of(e, lo, hi, cfg) == key]
            except (KeyError, TypeError, ValueError):
                span = []
        if not span:
            span = sorted({e for p in mine if (e := _ep(p)) is not None})
        out.append(Part(key, PART_NAMES.get(key, key.capitalize()), span[0], span[-1], len(mine),
                        mine[0]["rank"], mine[-1]["rank"]))
    return out


def replayable(promos: list[dict], cfg: dict) -> bool:
    """Every promo has flags.block, and cfg.scoring has the 3 variety keys."""
    sc = _d(_d(cfg).get("scoring"))
    if not promos or not all(isinstance(sc.get(k), (int, float)) for k in _VARIETY):
        return False
    return all(_d(p.get("flags")).get("block") in BLOCKS and isinstance(p.get("score"), (int, float))
               and isinstance(p.get("rank"), int) and _ep(p) is not None for p in promos)


def _counted(q: dict, placed: list[dict], sc: dict) -> tuple[float | None, tuple[str, ...]]:
    """q's counted value and the reasons, before the next promo of its part is placed."""
    ep = _ep(q)
    same = sum(1 for x in placed if _ep(x) == ep)
    cap = int(sc["max_per_episode_top10"])
    if len(placed) < 10 and same >= cap:
        return None, (f"episode {ep} already has {cap} promos in this part's first 10 places",)
    sep, stp = float(sc["same_episode_penalty"]), float(sc["same_type_penalty"])
    value = float(q["score"]) * (1 - sep) ** same
    reasons: list[str] = []
    if same:
        nth = _NTH.get(same + 1, fmt.ordinal(same + 1))
        times = _TIMES.get(same, f" {same} times")
        reasons.append(f"a {nth} promo from episode {ep}, {fmt.mult(1 - sep)}{times}")
    prev = placed[-1] if placed else None
    ptype = _d(prev.get("flags")).get("hook_type") if prev else None
    if ptype and _d(q.get("flags")).get("hook_type") == ptype:
        value *= 1 - stp
        reasons.append(f"opens on {TYPE_WORDS.get(ptype, TYPE_WORDS['other'])} like #{prev['rank']}, {fmt.mult(1 - stp)}")
    return value, tuple(reasons)


def replay(promos: list[dict], cfg: dict) -> dict[str, Placement]:
    """Keyed by promo_id. For each part, walk its promos in rank order; before each is placed, count every promo not
    yet placed in that part (spec 5.4)."""
    if not replayable(promos, cfg):
        return {}
    sc = _d(cfg["scoring"])
    out: dict[str, Placement] = {}
    for key in BLOCKS:
        mine = [p for p in _ranked(promos) if p["flags"]["block"] == key]
        placed: list[dict] = []
        for i, p in enumerate(mine):
            counted, reasons = _counted(p, placed, sc)
            rivals = []
            for q in mine[i + 1:]:
                if float(q["score"]) > float(p["score"]):
                    c, r = _counted(q, placed, sc)
                    rivals.append(Rival(q["rank"], float(q["score"]), c, r))
            out[p["promo_id"]] = Placement(p["rank"], key, float(p["score"]),
                                           float(p["score"]) if counted is None else counted,
                                           reasons, tuple(rivals), not placed)
            placed.append(p)
    return out


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _and(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def _rival_text(rivals: tuple[Rival, ...]) -> str:
    if not rivals:
        return ""
    named = sorted(sorted(rivals, key=lambda r: (-r.score, r.rank))[:2], key=lambda r: r.rank)
    ahead = _and([f"#{r.rank} ({fmt.score(r.score)})" for r in named])
    counted = [r for r in named if r.counted is not None]
    held = [r for r in named if r.counted is None]
    bits = []
    if counted:
        bits.append(f"which counted only {_and([fmt.score(r.counted) for r in counted])} at this point" if not held
                    else f"#{counted[0].rank} counted only {fmt.score(counted[0].counted)} at this point")
    detail = ". ".join(f"#{r.rank}: {'; '.join(r.reasons)}" for r in named if r.reasons)
    if held and not counted:
        return f" Posted ahead of {ahead}, which could not be placed yet ({detail})."
    if held:
        return (f" Posted ahead of {ahead}: {bits[0]}, and #{held[0].rank} could not be placed yet ({detail}).")
    return f" Posted ahead of {ahead}, {bits[0]} ({detail})." if detail else f" Posted ahead of {ahead}, {bits[0]}."


def why_sentence(pid: str, placements: dict, promos: list[dict], parts_: list[Part]) -> str:
    """Spec 5.4 'why_sentence' templates. "" when the promo was not replayed."""
    pl = placements.get(pid) if placements else None
    if pl is None:
        return ""
    part = next((x for x in parts_ if x.key == pl.part), None)
    rng = fmt.ep_range(part.first_ep, part.last_ep) if part else "this part"
    sc = fmt.score(pl.score)
    if pl.first_in_part:
        idx = parts_.index(part) if part in parts_ else 0
        if idx == 0:
            text = f"Posted first: the highest score in {rng} ({sc})." if pl.rank == 1 else \
                f"The highest score in {rng} ({sc})."
        else:
            first = parts_[0].first_ep
            before = fmt.ep_range(first, parts_[idx - 1].last_ep)
            scores = [float(p["score"]) for p in promos if isinstance(p.get("score"), (int, float))]
            if scores and pl.score >= max(scores):
                text = (f"Highest score of all {len(promos)} ({sc}), but it comes from {rng}. {_cap(before)} fill the "
                        f"first {pl.rank - 1} places, because a viewer who taps starts at episode {first}.")
            else:
                text = (f"The highest score in {rng} ({sc}). That part is posted after {before}, which fill the first "
                        f"{pl.rank - 1} places.")
    elif not pl.reasons:
        text = f"Next highest score left in {rng} ({sc})."
    else:
        text = f"Next in {rng}: score {sc}, counted as {fmt.score(pl.counted)} ({'; '.join(pl.reasons)})."
    return text + _rival_text(pl.rivals)


def _frac_words(x: float) -> str:
    words = {0.2: "fifth", 0.25: "quarter", 0.5: "half"}
    for k, w in words.items():
        if abs(float(x) - k) < 1e-9:
            return f"first {w}"
    return f"first {fmt.pct(x)}"


def _counts(data: dict, parts_: list[Part]) -> dict[str, int]:
    sb = _d(_d(_d(data.get("metrics")).get("editor_rubric")).get("slate_blocks"))
    if sb and all(isinstance(sb.get(p.key), int) for p in parts_):
        return {p.key: int(sb[p.key]) for p in parts_}
    return {p.key: p.n for p in parts_}


def rule_lines(data: dict, parts_: list[Part]) -> list[str]:
    """The rule box as markdown lines (spec 5.4). Every number is read from the run."""
    cfg = _d(data.get("cfg"))
    promos = data.get("promos") or []
    sc = _d(cfg.get("scoring"))
    slate = _d(cfg.get("slate"))
    lines = ["**How the order is set (plain code, no AI)**"]
    counts = _counts(data, parts_)
    rngs = [fmt.ep_range(p.first_ep, p.last_ep) for p in parts_]
    if len(parts_) >= 3:
        split = (f"So {rngs[0]} take the first {counts[parts_[0].key]} places, "
                 + ", ".join(f"{r} the next {counts[p.key]}" for r, p in zip(rngs[1:-1], parts_[1:-1]))
                 + f" and {rngs[-1]} the last {counts[parts_[-1].key]}.")
    elif len(parts_) == 2:
        split = (f"So {rngs[0]} take the first {counts[parts_[0].key]} places and {rngs[1]} the last "
                 f"{counts[parts_[1].key]}.")
    elif parts_:
        split = f"So all {counts[parts_[0].key]} places go to {rngs[0]}."
    else:
        split = ""
    paywall = _d(_d(data.get("run_log")).get("args")).get("paywall") or slate.get("paywall")
    if paywall and parts_:
        first = (f"1. **Free episodes first.** Episodes 1–{int(paywall) - 1} are free, so they take the first "
                 f"{counts[parts_[0].key]} places.")
        rest = [f"{r} the {'last' if i == len(parts_) - 2 else 'next'} {counts[p.key]}"
                for i, (r, p) in enumerate(zip(rngs[1:], parts_[1:]))]
        if rest:
            first += f" Then {_and(rest)}."
    else:
        first = f"1. **Opening first.** A viewer who taps a promo starts the show at episode {parts_[0].first_ep if parts_ else 1}."
        first += f" {split}" if split else ""
    shares = _d(slate.get("shares"))
    if shares and all(isinstance(shares.get(b), (int, float)) for b in BLOCKS):
        share_txt = " / ".join(fmt.pct(shares[b])[:-1] for b in BLOCKS) + "%"
        dd = _d(data.get("dramaad"))
        foot = _d(dd.get("share_footage_seconds"))
        if dd.get("ads") and foot and all(isinstance(foot.get(b), (int, float)) for b in BLOCKS):
            oe, me = slate.get("opening_end"), slate.get("middle_end")
            mid = f"the next {fmt.pct(float(me) - float(oe))}" if isinstance(oe, (int, float)) and isinstance(
                me, (int, float)) else "the middle"
            late = "the second half" if isinstance(me, (int, float)) and abs(float(me) - 0.5) < 1e-9 else "the rest"
            opening = _frac_words(oe) if isinstance(oe, (int, float)) else "opening"
            first += (f" The {share_txt} split copies professional editors: in {fmt.num(dd['ads'])} ads from "
                      f"{fmt.num(dd.get('dramas', 0))} dramas, {fmt.pct(foot['opening'])} of the footage came from the "
                      f"{opening} of the show, {fmt.pct(foot['middle'])} from {mid} and {fmt.pct(foot['late'])} from "
                      f"{late}.")
        else:
            first += (f" The {share_txt} split copies where professional editors take ad footage from (DramaAD "
                      "dataset; see Method).")
    lines.append(first)
    if all(isinstance(sc.get(k), (int, float)) for k in _VARIETY):
        lines.append(f"2. **Best first inside each part, with variety.** A promo counts "
                     f"{fmt.mult(1 - float(sc['same_episode_penalty']))} for each promo already taken from its episode, "
                     f"and {fmt.mult(1 - float(sc['same_type_penalty']))} if it opens on the same kind of moment as the "
                     f"promo before it. At most {int(sc['max_per_episode_top10'])} promos per episode in a part's first "
                     "10 places.")
    E = len(data.get("episodes_list") or [])
    if 0 < E < 3:
        lines.append(f"With {E} episodes, parts are coarse.")
    scored = [p for p in promos if isinstance(p.get("score"), (int, float)) and isinstance(p.get("rank"), int)]
    if scored:
        best = max(scored, key=lambda p: (p["score"], -p["rank"]))
        if best["rank"] != 1:
            lines.append(f"So # is the order to post in, not a quality rank: #{best['rank']} has the highest score of all "
                         f"{len(promos)} ({fmt.score(best['score'])}) but comes from episode {_ep(best)}.")
        else:
            lines.append("So # is the order to post in, not a quality rank.")
    return lines
