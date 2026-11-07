"""Which episodes the promos come from, and in what order to post them.

Evidence (the DramaAD dataset released with arXiv 2507.02790, EMNLP 2025 Industry track; 501 ads cut by professional editors from 28 short dramas of
72-109 episodes, measured by scripts/benchmark_dramaad.py):
  - 55% of ad footage comes from the first 20% of episodes, 33% from 20-50%, 12% from the second half;
  - 55% of ads OPEN on footage from the first 20% of episodes (40% from the first 10%);
  - but no drama's ads stay entirely in its opening: editors also use mid-show peaks.
A promo is an acquisition ad: the viewer taps and lands on episode 1, so the promise should be reached soon.

So the slate copies the professional shape instead of letting raw scores decide: a quota per block of the show
(opening / middle / late, shares in config.yaml -> slate), best-first inside each block, opening block posted first.
With --paywall N the opening block is exactly the free episodes (1 .. N-1).
"""
from __future__ import annotations

from .beat_scoring import ScoredShow
from .models import Window

BLOCKS = ("opening", "middle", "late")


def episode_range(ctx: ScoredShow) -> tuple[int, int]:
    nums = [e.number for e in ctx.episodes]
    return min(nums), max(nums)


def position(episode: int, lo: int, hi: int) -> float:
    """0.0 = first episode, 1.0 = last episode."""
    return (episode - lo) / max(hi - lo, 1)


def block_of(episode: int, lo: int, hi: int, cfg: dict) -> str:
    s = cfg["slate"]
    pos = position(episode, lo, hi)
    paywall = s.get("paywall")
    if paywall:
        if episode < int(paywall):
            return "opening"
        return "middle" if pos < s["middle_end"] else "late"
    if pos < s["opening_end"]:
        return "opening"
    return "middle" if pos < s["middle_end"] else "late"


def minutes_into_show(ctx: ScoredShow) -> dict[int, float]:
    """Seconds of watching before each episode starts (episode 1 starts at 0)."""
    out, t = {}, 0.0
    for ep in sorted(ctx.episodes, key=lambda e: e.number):
        out[ep.number] = t
        t += ep.duration
    return out


def quotas(n: int, cfg: dict) -> dict[str, int]:
    """Largest-remainder rounding of n * share per block."""
    shares = cfg["slate"]["shares"]
    tot = sum(float(shares[b]) for b in BLOCKS)
    raw = {b: n * float(shares[b]) / tot for b in BLOCKS}
    q = {b: int(raw[b]) for b in BLOCKS}
    for b in sorted(BLOCKS, key=lambda b: raw[b] - q[b], reverse=True)[: n - sum(q.values())]:
        q[b] += 1
    return q


def build(windows: list[Window], cfg: dict, n: int, order_fn) -> list[Window]:
    """Fill each block's quota best-first (order_fn = rank.diversity_order); a block that runs short passes its
    unused places on to the next block, then to any block. Returned in posting order: opening, middle, late."""
    q = quotas(n, cfg)
    by_block = {b: [w for w in windows if w.flags.get("block") == b] for b in BLOCKS}
    chosen: list[Window] = []
    carry = 0
    for b in BLOCKS:
        want = q[b] + carry
        picked = order_fn(by_block[b], cfg, want) if want > 0 else []
        carry = want - len(picked)
        chosen += picked
    if carry > 0:  # every block ran short (a tiny show): best of whatever is left
        rest = [w for w in windows if w not in chosen]
        chosen += order_fn(rest, cfg, carry)
    return chosen


def shape(windows: list[Window]) -> dict[str, int]:
    return {b: sum(1 for w in windows if w.flags.get("block") == b) for b in BLOCKS}
