import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from promo_finder.beat_scoring import BeatScoresOut, _merge, score_show
from promo_finder.beats import build_beats
from promo_finder.features import heuristic_scores
from promo_finder.llm import LLM, LLMError, RunLog, parse_model_string
from promo_finder.llm.base import _extract_json
from promo_finder.llm.openai_compat import _loosen


class Out(BaseModel):
    beats: list[dict]


class FakeLLM(LLM):
    """Scripted backend: returns queued texts, records calls."""

    def __init__(self, texts, cache_dir=None):
        super().__init__("fake", "m", {"rpm": 10000, "max_retries": 0}, cache_dir, RunLog())
        self.texts, self.calls = list(texts), 0

    def _raw_call(self, system, user, schema):
        self.calls += 1
        return self.texts.pop(0), 10, 5

    def list_models(self):
        return ["m"]


def test_parse_model_string():
    assert parse_model_string("none") is None
    assert parse_model_string("gemini/gemini-2.5-flash") == ("gemini", "gemini-2.5-flash")
    with pytest.raises(ValueError):
        parse_model_string("gemini")


def test_extract_json_tolerates_fences_and_prose():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('Sure! {"a": [1,2]} done') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        _extract_json("nothing here")


def test_repair_retry_then_success(tmp_path):
    llm = FakeLLM(['{"beats": "not a list"}', '{"beats": [{"beat_id": "b01"}]}'], cache_dir=tmp_path)
    out = llm.complete_json("sys", "user", Out, "t", "x")
    assert out.beats == [{"beat_id": "b01"}] and llm.calls == 2
    rec = llm.log.calls[-1]
    assert rec.repaired and rec.attempts == 2
    # second identical call is served from cache
    llm2 = FakeLLM([], cache_dir=tmp_path)
    assert llm2.complete_json("sys", "user", Out, "t", "x").beats == [{"beat_id": "b01"}]
    assert llm2.calls == 0 and llm2.log.calls[-1].cached


def test_gives_up_after_repair_fails(tmp_path):
    llm = FakeLLM(["garbage", "still garbage"])
    with pytest.raises(LLMError):
        llm.complete_json("sys", "user", Out, "t", "x")


def test_merge_handles_unknown_and_missing_ids(episodes):
    ep = episodes[0]
    beats = build_beats(ep)
    heur = heuristic_scores(ep, beats)
    out = BeatScoresOut.model_validate({"beats": [
        {"beat_id": "b01", "hook": 9, "cliff": -1, "beat_type": "SLAP", "unresolved_refs": ["Tom"]},
        {"beat_id": "b99", "hook": 1, "cliff": 1},
    ]})
    merged, info = _merge(ep, beats, out, heur)
    assert merged["b01"].hook == 3 and merged["b01"].cliff == 0 and merged["b01"].beat_type == "slap"
    assert merged["b01"].source == "llm"
    assert info["unknown_ids"] == ["b99"]
    assert "b02" in info["missing_ids_filled_from_heuristic"] and merged["b02"].source == "heuristic"


def test_score_show_falls_back_per_episode_on_failure(episodes, cfg):
    # first episode: valid JSON; second: invalid twice -> heuristic fallback recorded
    beats1 = build_beats(episodes[0])
    good = json.dumps({"beats": [{"beat_id": b.beat_id, "hook": 1, "cliff": 1, "beat_type": "line"} for b in beats1]})
    llm = FakeLLM([good, "nope", "nope"])
    cfg = dict(cfg)
    cfg["llm"] = {"concurrency": 1}
    ctx = score_show(episodes, llm, cfg, None)
    assert ctx.llm_by_ep[1] and not ctx.llm_by_ep[2]
    assert 2 in ctx.llm_failures and ctx.has_llm
    assert ctx.scores(2)["b01"].source == "heuristic"


def test_loosen_strips_constraints_and_inlines_refs():
    from promo_finder.models import ColdViewer
    s = _loosen(ColdViewer.model_json_schema())
    assert "minimum" not in json.dumps(s) and "$ref" not in json.dumps(s)


class ForbiddenLLM(FakeLLM):
    def _raw_call(self, system, user, schema):
        class Resp:  # mimics an SDK error carrying a status code and a Google-style body
            status_code = 403
            body = [{"error": {"code": 403, "message": "The Gemini API free tier is not available for this project."}}]

        e = RuntimeError("Error code: 403")
        e.status_code, e.body = Resp.status_code, Resp.body
        raise e


def test_permission_errors_are_fatal_and_carry_the_provider_message(episodes, cfg):
    llm = ForbiddenLLM([])
    with pytest.raises(LLMError) as ei:
        llm.complete_json("s", "u", Out, "t", "x")
    assert ei.value.fatal and "free tier is not available" in str(ei.value)
    assert llm.log.calls[-1].error.startswith("403")
    cfg = dict(cfg); cfg["llm"] = {"concurrency": 1}
    with pytest.raises(LLMError):  # score_show must abort, not fall back episode by episode
        score_show(episodes, ForbiddenLLM([]), cfg, None)


def test_out_of_credit_is_fatal_but_inflight_reservation_is_retried():
    class Broke(FakeLLM):
        def _raw_call(self, system, user, schema):
            e = RuntimeError("Error code: 402"); e.status_code = 402
            e.body = {"error": {"code": 402, "message": "This request requires more credits, or fewer max_tokens."}}
            raise e

    with pytest.raises(LLMError) as ei:
        Broke([]).complete_json("s", "u", Out, "t", "x")
    assert ei.value.fatal and "more credits" in str(ei.value)

    class Flaky(FakeLLM):
        n = 0
        def _raw_call(self, system, user, schema):
            Flaky.n += 1
            if Flaky.n == 1:
                e = RuntimeError("402"); e.status_code = 402
                e.body = {"error": {"message": "would exceed your available credits given your current in-flight requests"}}
                raise e
            return '{"beats": []}', 1, 1

    llm = Flaky([]); llm.max_retries = 2
    import promo_finder.llm.base as b; b.time.sleep = lambda s: None
    assert llm.complete_json("s", "u", Out, "t", "y").beats == [] and Flaky.n == 2


def test_openrouter_key_pool_rotates_on_daily_cap(monkeypatch):
    from promo_finder.llm.openai_compat import OpenAICompatLLM
    monkeypatch.setenv("OPENROUTER_API_KEY", "k1"); monkeypatch.setenv("OPENROUTER_API_KEY2", "k2")
    llm = OpenAICompatLLM("openrouter", "x/y:free", {"rpm": 100000, "max_retries": 0}, None, RunLog())
    assert llm.keys == ["k1", "k2"]
    assert llm._rotate_key("free-models-per-day limit exceeded") and llm.key_idx == 1
    assert not llm._rotate_key("again")
    assert any("rotated" in (c.error or "") for c in llm.log.calls)


def test_anthropic_adapter_parses_structured_output(monkeypatch):
    import anthropic
    from promo_finder.llm.anthropic_client import AnthropicLLM

    class Usage:
        input_tokens, output_tokens = 11, 7

    class Resp:
        stop_reason = "end_turn"
        usage = Usage()
        parsed_output = Out(beats=[{"beat_id": "b01"}])
        content = []

    class Messages:
        def parse(self, **kw):
            assert kw["output_format"] is Out and kw["model"] == "model-x"
            return Resp()

    class Client:
        def __init__(self, **kw):
            self.messages = Messages()

    monkeypatch.setattr(anthropic, "Anthropic", Client)
    llm = AnthropicLLM("anthropic", "model-x", {"rpm": 100000, "max_retries": 0}, None, RunLog())
    out = llm.complete_json("s", "u", Out, "t", "x")
    assert out.beats == [{"beat_id": "b01"}] and llm.log.calls[-1].input_tokens == 11


def test_alias_merging_is_conservative(episodes):
    from promo_finder.beat_scoring import build_aliases
    llm = FakeLLM(['{"aliases": {"LADY MARA": "MARA", "STRANGER": "TOM", "MARA": "MARA"}, "notes": "n"}'])
    # give the fixture a LADY MARA label so the suggestion refers to a real speaker
    eps = [e.model_copy(deep=True) for e in episodes]
    eps[0].events[1].speaker = "LADY MARA"
    aliases, notes = build_aliases(eps, llm, None)
    assert aliases == {"LADY MARA": "MARA"}
    assert "STRANGER" in notes and "not applied" in notes


def test_batch_missing_episode_is_rescored_alone(episodes, cfg):
    beats1 = build_beats(episodes[0]); beats2 = build_beats(episodes[1])
    batch = json.dumps({"episodes": [{"episode": 1, "beats": [{"beat_id": b.beat_id, "hook": 1 + (i == 0), "cliff": 1 + (i == len(beats1) - 1), "beat_type": "line"} for i, b in enumerate(beats1)]}]})
    single = json.dumps({"beats": [{"beat_id": b.beat_id, "hook": 2, "cliff": 2, "beat_type": "line"} for b in beats2]})
    llm = FakeLLM([batch, single])
    cfg = dict(cfg); cfg["llm"] = {"concurrency": 1, "episodes_per_call": 5}
    ctx = score_show(episodes, llm, cfg, None)
    assert llm.calls == 2 and not ctx.llm_failures
    assert ctx.scores(2)["b01"].source == "llm" and ctx.scores(2)["b01"].hook == 2
    assert "re-scored alone" in ctx.id_issues[2]["note"]


def _varied(beats, top=3):
    return [{"beat_id": b.beat_id, "hook": top if i == 0 else 1, "cliff": top if i == len(beats) - 1 else 0, "beat_type": "line"} for i, b in enumerate(beats)]


def test_flat_batch_answer_is_re_asked_alone(episodes, cfg):
    """A small model sometimes fills a whole episode with zeros inside a batch. The tool re-asks that episode alone."""
    beats1 = build_beats(episodes[0]); beats2 = build_beats(episodes[1])
    batch = json.dumps({"episodes": [
        {"episode": 1, "beats": _varied(beats1)},
        {"episode": 2, "beats": [{"beat_id": b.beat_id, "hook": 0, "cliff": 0, "beat_type": "line"} for b in beats2]}]})
    retry = json.dumps({"beats": _varied(beats2)})
    llm = FakeLLM([batch, retry])
    cfg = dict(cfg); cfg["llm"] = {"concurrency": 1, "episodes_per_call": 5}
    ctx = score_show(episodes, llm, cfg, None)
    assert llm.calls == 2
    assert max(s.hook for s in ctx.scores(2).values()) == 3
    assert "flat answer" in ctx.id_issues[2]["note"] and "fixed" in ctx.id_issues[2]["note"]


def test_flat_answer_confirmed_on_retry_is_kept_and_flagged(episodes, cfg):
    beats1 = build_beats(episodes[0])
    flat = json.dumps({"beats": [{"beat_id": b.beat_id, "hook": 1, "cliff": 0, "beat_type": "line"} for b in beats1]})
    llm = FakeLLM([flat, flat])
    cfg = dict(cfg); cfg["llm"] = {"concurrency": 1, "episodes_per_call": 1}
    ctx = score_show(episodes[:1], llm, cfg, None)
    assert llm.calls == 2 and not ctx.llm_failures
    assert ctx.scores(1)[beats1[0].beat_id].source == "llm"
    assert "kept: the model means it" in ctx.id_issues[1]["note"]


def test_votes_average_independent_readings(episodes, cfg):
    """--votes 2: two readings per episode (the second with a different prompt), averaged."""
    beats = build_beats(episodes[0])
    r1 = json.dumps({"beats": [{"beat_id": b.beat_id, "hook": 3 if i == 0 else 0, "cliff": 2 if i == len(beats) - 1 else 0, "beat_type": "threat", "role": "escalation"} for i, b in enumerate(beats)]})
    r2 = json.dumps({"beats": [{"beat_id": b.beat_id, "hook": 2 if i == 0 else 0, "cliff": 3 if i == len(beats) - 1 else 0, "beat_type": "line", "role": "escalation"} for i, b in enumerate(beats)]})
    llm = FakeLLM([r1, r2])
    cfg = dict(cfg); cfg["llm"] = {"concurrency": 1, "episodes_per_call": 1}
    ctx = score_show(episodes[:1], llm, cfg, None, votes=2)
    first, last = beats[0].beat_id, beats[-1].beat_id
    assert llm.calls == 2
    assert ctx.scores(1)[first].hook == 2.5 and ctx.scores(1)[last].cliff == 2.5
    assert ctx.scores(1)[first].role == "escalation"
