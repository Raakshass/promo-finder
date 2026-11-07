"""Editor's rubric, promo-ability and the blind A/B test — synthetic data only."""
import yaml
from pathlib import Path

from pydantic import BaseModel

from promo_finder import rubric as rb
from promo_finder.judge import ab_test
from promo_finder.models import Beat, BeatScore, ColdViewer, Segment, Window

CFG = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text())


def _beats(spec):
    """spec: list of (speaker, role, intensity, t0, t1)"""
    beats, scores = [], []
    for i, (spk, role, inten, t0, t1) in enumerate(spec):
        bid = f"b{i:02d}"
        beats.append(Beat(beat_id=bid, global_id=f"e001-{bid}", episode=1, idx=i, kind="dialogue", t0=t0, t1=t1, speaker=spk, text="x"))
        scores.append(BeatScore(beat_id=bid, hook=2, cliff=2, role=role, intensity=inten, beat_type="line", source="llm"))
    return beats, scores


def _win():
    return Window(cand_id="c", segments=[Segment(episode=1, start_s=0, end_s=30)], hook_beat="e001-b00", cliff_beat="e001-b05", beat_ids=[], duration_s=30)


def test_payoff_after_last_push_fails_ends_open():
    beats, sc = _beats([("A", "escalation", 2, 0, 3), ("B", "escalation", 2, 3, 6), ("A", "turn", 3, 6, 9),
                        ("B", "payoff", 1, 9, 12), ("A", "none", 1, 12, 15), ("B", "none", 1, 15, 18)])
    r = rb.code_rubric(_win(), beats, sc, sc[0], sc[-1], CFG)
    assert r["ends_open"] is False


def test_payoff_before_the_last_push_still_ends_open():
    beats, sc = _beats([("A", "escalation", 2, 0, 3), ("B", "payoff", 1, 3, 6), ("A", "escalation", 2, 6, 9),
                        ("B", "escalation", 2, 9, 12), ("A", "turn", 3, 12, 15), ("B", "escalation", 3, 15, 18)])
    r = rb.code_rubric(_win(), beats, sc, sc[0], sc[-1], CFG)
    assert r["ends_open"] is True and r["builds"] is True and r["opens_on_conflict"] is True


def test_cast_dead_air_and_sag():
    beats, sc = _beats([("A", "escalation", 3, 0, 3), ("B", "escalation", 3, 3, 6), ("C", "escalation", 3, 6, 9),
                        ("D", "none", 0, 9, 12), ("A", "none", 0, 20, 23), ("B", "none", 0, 23, 26)])
    r = rb.code_rubric(_win(), beats, sc, sc[0], sc[-1], CFG)
    assert r["small_cast"] is False      # 4 speakers
    assert r["no_dead_air"] is False     # 8 s gap
    assert r["builds"] is False          # hot start, flat end


def test_multiplier_and_judge_override():
    rub = {k: True for k in rb.CODE_CHECKS}
    assert rb.multiplier(rub, CFG) == 1.0
    cv = ColdViewer(comprehensibility=4, what_happened="x", want_next="y", why_it_works="z", would_tap=4,
                    stakes="", root_for="her", resolved_inside=True, opening_conflict=True)
    j = rb.judge_rubric(rub, cv)
    assert j["ends_open"] is False and j["stakes_clear"] is False and j["rootable"] is True
    pen = CFG["rubric"]["penalties"]
    assert abs(rb.multiplier(j, CFG) - pen["ends_open"] * pen["stakes_clear"]) < 1e-9


def test_blocks_and_quotas_follow_professional_shape():
    from promo_finder import slate as sl
    assert sl.block_of(1, 1, 50, CFG) == "opening" and sl.block_of(10, 1, 50, CFG) == "opening"
    assert sl.block_of(11, 1, 50, CFG) == "middle" and sl.block_of(25, 1, 50, CFG) == "middle"
    assert sl.block_of(26, 1, 50, CFG) == "late" and sl.block_of(50, 1, 50, CFG) == "late"
    assert [sl.block_of(e, 1, 10, CFG) for e in range(1, 11)] == ["opening"] * 2 + ["middle"] * 3 + ["late"] * 5
    assert sl.quotas(20, CFG) == {"opening": 11, "middle": 7, "late": 2}
    cfg = {**CFG, "slate": {**CFG["slate"], "paywall": 6}}
    assert sl.block_of(5, 1, 50, cfg) == "opening" and sl.block_of(6, 1, 50, cfg) == "middle"


def test_slate_fills_quotas_and_passes_shortfall_on():
    from promo_finder import slate as sl
    from promo_finder.models import Segment, Window
    def w(i, block, score):
        x = Window(cand_id=f"c{i}", segments=[Segment(episode=i, start_s=0, end_s=30)], hook_beat=f"h{i}", cliff_beat=f"k{i}", beat_ids=[], duration_s=30)
        x.flags = {"block": block, "episode": i, "hook_type": "line"}; x.score = score
        return x
    pool = [w(i, "opening", 0.5) for i in range(1, 4)] + [w(i, "middle", 0.9) for i in range(10, 30)] + [w(i, "late", 0.95) for i in range(40, 45)]
    order = lambda ws, cfg, n: sorted(ws, key=lambda x: -x.score)[:n]  # noqa: E731
    out = sl.build(pool, CFG, 20, order)
    assert [x.flags["block"] for x in out[:3]] == ["opening"] * 3   # opening posted first even with lower scores
    assert sl.shape(out) == {"opening": 3, "middle": 15, "late": 2}  # opening ran short by 8: middle takes them


def _promo(pid, ep, start):
    return {"promo_id": pid, "duration_s": 30, "segments": [{"episode": ep, "start_s": start, "end_s": start + 30}],
            "lines": [{"kind": "dialogue", "speaker": "A", "text": f"{pid} line"}]}


class _FakeLLM:
    """Answers the A/B prompt with a fixed policy."""
    def __init__(self, policy):
        self.policy = policy

    def complete_json(self, system, user, schema, stage, tag, raw_path=None):
        import re
        pairs = re.split(r"^PAIR (\d+)$", user, flags=re.M)
        items = []
        for num, body in zip(pairs[1::2], pairs[2::2]):
            first = re.search(r"CLIP 1 .*?\n(.*?)\n\nCLIP 2", body, flags=re.S).group(1)
            items.append({"pair": int(num), "winner": self.policy(first), "reason": "fake"})
        return schema(pairs=items)


def test_ab_position_bias_becomes_a_tie_and_real_preference_wins():
    a = [_promo("A1", 1, 0), _promo("A2", 2, 0), _promo("S", 3, 0)]
    b = [_promo("B1", 4, 0), _promo("B2", 5, 0), _promo("S", 3, 0)]
    always_first = ab_test(a, b, _FakeLLM(lambda first: "1"), None, k=3)
    assert always_first["ties"] == 2 and always_first["same_clip"] == 1 and always_first["a_wins"] == 0
    prefers_b = ab_test(a, b, _FakeLLM(lambda first: "1" if "B" in first else "2"), None, k=3)
    assert prefers_b["b_wins"] == 2 and prefers_b["position_consistency"] == 1.0


def test_promo_flags_accept_decimal_values():
    """minutes_into_show / position are floats; a too-narrow flags type once crashed the final write."""
    from promo_finder.models import Promo, PromoLine, PromoSegment
    p = Promo(promo_id="P01", rank=1, score=0.9, segments=[PromoSegment(episode=1, start="00:00.00", end="00:30.00", start_s=0, end_s=30)],
              duration_s=30, hook=PromoLine(beat_id="e001-b01", at_s=0.2, text="x", score=3),
              cliffhanger=PromoLine(beat_id="e001-b09", at_s=29, text="y", score=3), subscores={},
              flags={"minutes_into_show": 4.5, "position": 0.123, "block": "opening", "spoiler_risk": False})
    assert p.flags["minutes_into_show"] == 4.5


def test_promise_ledger_validates_episodes_and_computes_minutes(episodes):
    from promo_finder import ledger
    from promo_finder.beat_scoring import score_show
    from promo_finder.models import Promo, PromoLine, PromoSegment, ColdViewer

    class FakeLedgerLLM:
        def __init__(self, items): self.items = items
        def complete_json(self, system, user, schema, stage, tag, raw_path=None):
            return schema(items=self.items)

    ctx = score_show(episodes, None, CFG, None)  # two synthetic episodes
    d1 = ctx.episode(1).duration
    def promo(pid, ep, start, end, want):
        return Promo(promo_id=pid, rank=1, score=0.9, segments=[PromoSegment(episode=ep, start="", end="", start_s=start, end_s=end)],
                     duration_s=end - start, hook=PromoLine(beat_id="x", at_s=0, text="h", score=3),
                     cliffhanger=PromoLine(beat_id="y", at_s=end - start, text="Will she go?", score=3), subscores={},
                     cold_viewer=ColdViewer(comprehensibility=4, what_happened="w", want_next=want, why_it_works="z", would_tap=4))
    ps = [promo("P01", 1, 10, 40, "Will she marry the stranger?"), promo("P02", 2, 5, 30, "Who is he?"), promo("P03", 1, 0, 25, "Why?")]
    items = [ledger.LedgerItem(promo_id="P01", answered_in_episode=2, answer="she does", confidence="high"),
             ledger.LedgerItem(promo_id="P02", answered_in_episode=1, answer="before the clip: impossible"),
             ledger.LedgerItem(promo_id="P03", answered_in_episode=None, answer="never")]
    rows = ledger.build(ps, ctx, FakeLedgerLLM(items), None, paywall=2)
    assert rows["P01"]["question"] == "Will she marry the stranger?"
    assert rows["P01"]["reached_at_min"] == round(10 / 60, 1)
    assert rows["P01"]["answered_in_episode"] == 2 and rows["P01"]["answer_is_free"] is False
    assert rows["P01"]["minutes_after_clip"] == round((d1 - 40) / 60, 1)
    assert rows["P02"]["status"] == "rejected"           # an answer cannot come before the question
    assert rows["P03"]["status"] == "never answered"     # fake-hook warning
    s = ledger.summary(rows)
    assert s["answered"] == 1 and s["never_answered"] == 1 and s["unchecked_or_rejected"] == 1
