from promo_finder.beat_scoring import ScoredShow
from promo_finder.beats import build_beats, cut_set
from promo_finder.candidates import boundary_windows, windows_for_episode
from promo_finder.features import heuristic_scores
from promo_finder.rank import rank_windows


def _ctx(episodes):
    beats = {e.number: build_beats(e) for e in episodes}
    heur = {e.number: heuristic_scores(e, beats[e.number]) for e in episodes}
    return ScoredShow(episodes, beats, {e.number: {} for e in episodes}, heur)


def test_beats_exclude_camera_and_other(episodes):
    beats = build_beats(episodes[0])
    assert all(b.kind in ("dialogue", "action", "sfx") for b in beats)
    assert beats[0].beat_id == "b01" and beats[0].global_id == "e001-b01"
    assert any(b.speaker == "LENA" and b.emotions == ["Gloating"] for b in beats)


def test_cut_set_contains_camera_boundaries(episodes):
    cuts = cut_set(episodes[0])
    assert 3.0 in cuts and 8.0 in cuts and 40.0 in cuts


def test_windows_respect_invariants(episodes, cfg):
    ctx = _ctx(episodes)
    p = cfg["promo"]
    for ep in episodes:
        ws = windows_for_episode(ep, ctx.beats_by_ep[ep.number], ctx.heur_by_ep[ep.number], cfg)
        assert ws, f"episode {ep.number} produced no windows"
        for w in ws:
            assert p["min_duration_s"] <= w.duration_s <= p["max_duration_s"] + 1e-6
            start = w.segments[0].start_s
            hook = ctx.beat(w.hook_beat)
            assert 0 <= hook.t0 - start <= p["hook_window_s"]
            assert start in cut_set(ep) or start == hook.t0
            assert ctx.beat(w.cliff_beat).kind != "sfx"
            for e in ep.events:  # no dialogue line overhangs a boundary
                if e.kind == "dialogue":
                    for b in (start, w.segments[-1].end_s):
                        assert not (e.t0 < b - p["straddle_tolerance_s"] and e.t1 > b + p["straddle_tolerance_s"])


def test_boundary_windows_split_into_two_ordered_segments(episodes, cfg):
    ctx = _ctx(episodes)
    a, b = episodes
    ws = boundary_windows(a, ctx.beats_by_ep[1], ctx.heur_by_ep[1], b, ctx.beats_by_ep[2], ctx.heur_by_ep[2], cfg)
    assert ws
    for w in ws:
        assert [s.episode for s in w.segments] == [1, 2]
        assert w.segments[0].end_s == a.duration and w.segments[1].start_s == 0.0
        assert abs((w.segments[0].duration_s + w.segments[1].duration_s) - w.duration_s) < 0.01


def test_rank_produces_scored_unique_windows(episodes, cfg):
    ctx = _ctx(episodes)
    ranked, all_w = rank_windows(ctx, cfg, "heuristic", top_n=5)
    assert ranked and all(0 <= w.score <= 1 for w in ranked)
    assert len({w.cand_id for w in ranked}) == len(ranked)
    assert all(set(w.subscores) >= {"hook", "cliff", "standalone", "density"} for w in ranked)
    assert all("spoiler_risk" in w.flags for w in ranked)


def test_continuation_lines_are_never_hooks():
    from promo_finder.candidates import _continuation
    assert _continuation('"...and then she ran for the door"') and _continuation("…and then?") and _continuation("but why")
    assert not _continuation('"No! That letter was never in my desk!"') and not _continuation("Because I am marrying your sister.")


def test_interrupted_line_counts_as_cliff(episodes):
    from promo_finder.beats import build_beats
    from promo_finder.features import heuristic_cliff
    ep = episodes[0]
    b = build_beats(ep)[0].model_copy(update={"text": "You let me believe—", "t1": 5.0})
    assert heuristic_cliff(b, ep, False) >= 1


def test_romantic_resolution_is_capped_as_cliff(episodes, cfg):
    from promo_finder.beat_scoring import ScoredShow
    from promo_finder.beats import build_beats
    from promo_finder.features import heuristic_scores
    from promo_finder.models import BeatScore
    from promo_finder.rank import rank_windows
    ep = episodes[1]
    beats = build_beats(ep)
    heur = heuristic_scores(ep, beats)
    llm = {b.beat_id: BeatScore(beat_id=b.beat_id, hook=1, cliff=1, beat_type="line", source="llm") for b in beats}
    last = beats[-1]  # "Yes." — a romantic resolution
    llm[last.beat_id] = BeatScore(beat_id=last.beat_id, hook=1, cliff=3, beat_type="romance", source="llm")
    ctx = ScoredShow([ep], {ep.number: beats}, {ep.number: llm}, {ep.number: heur})
    ranked, all_w = rank_windows(ctx, cfg, "llm", top_n=10)
    capped = [w for w in all_w if w.cliff_beat == last.global_id]
    assert capped and all(w.flags.get("romance_cap") and w.subscores["cliff"] <= 0.67 for w in capped)


def test_nms_suppresses_clips_sharing_a_key_beat(cfg):
    from promo_finder.models import Segment, Window
    from promo_finder.rank import nms
    a = Window(cand_id="a", segments=[Segment(episode=1, start_s=0, end_s=30)], hook_beat="e001-b01", cliff_beat="e001-b09", beat_ids=[], duration_s=30, score=0.9)
    b = Window(cand_id="b", segments=[Segment(episode=2, start_s=10, end_s=40)], hook_beat="e002-b03", cliff_beat="e001-b01", beat_ids=[], duration_s=30, score=0.8)  # ends on a's opening line
    c = Window(cand_id="c", segments=[Segment(episode=2, start_s=50, end_s=80)], hook_beat="e002-b20", cliff_beat="e002-b28", beat_ids=[], duration_s=30, score=0.7)
    assert [w.cand_id for w in nms([a, b, c], cfg)] == ["a", "c"]
