"""Provider-agnostic JSON completion with cache, throttle and retries.

`PROMO_MODEL=provider/model` selects the backend:
  openai-compatible: gemini | groq | openrouter | openai | ollama   (one adapter, different base_url)
  anthropic                                                        (native SDK, structured outputs)
  none                                                             (no model; callers use the heuristic twin)
The model only ever returns JSON validated against a pydantic schema. Timestamps never pass through here.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# USD per 1M tokens (input, output). Free tiers cost 0 but the paid rate is what scale would cost.
PRICES: dict[str, tuple[float, float]] = {
    "gemini/gemini-2.5-flash": (0.30, 2.50),
    "gemini/gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini/gemini-2.5-pro": (1.25, 10.00),
    "groq/llama-3.3-70b-versatile": (0.59, 0.79),
    "groq/llama-3.1-8b-instant": (0.05, 0.08),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "openrouter/google/gemini-2.5-flash": (0.30, 2.50),
    "openrouter/google/gemini-2.5-flash-lite": (0.10, 0.40),
    "openrouter/meta-llama/llama-3.3-70b-instruct": (0.59, 0.79),
    "openrouter/deepseek/deepseek-chat-v3-0324": (0.29, 1.14),
    "openai/gpt-4.1": (2.00, 8.00),
}


class LLMError(RuntimeError):
    """fatal=True means retrying or falling back per-episode is pointless (bad key, no access, no such model)."""

    def __init__(self, msg: str, fatal: bool = False):
        super().__init__(msg)
        self.fatal = fatal


@dataclass
class CallRecord:
    stage: str
    tag: str
    provider: str
    model: str
    prompt_sha: str
    cached: bool
    input_tokens: int = 0
    output_tokens: int = 0
    attempts: int = 1
    repaired: bool = False
    error: str | None = None
    seconds: float = 0.0
    response_text: str = ""

    def est_cost_usd(self) -> float:
        p = PRICES.get(f"{self.provider}/{self.model}")
        if not p:
            return 0.0
        return self.input_tokens / 1e6 * p[0] + self.output_tokens / 1e6 * p[1]


@dataclass
class RunLog:
    calls: list[CallRecord] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, rec: CallRecord) -> None:
        with self.lock:
            self.calls.append(rec)

    def summary(self) -> dict:
        live = [c for c in self.calls if not c.cached]
        return {
            "calls_total": len(self.calls),
            "calls_live": len(live),
            "calls_cached": len(self.calls) - len(live),
            "input_tokens": sum(c.input_tokens for c in self.calls),
            "output_tokens": sum(c.output_tokens for c in self.calls),
            "est_cost_usd_at_paid_rates": round(sum(c.est_cost_usd() for c in self.calls), 4),
            "errors": [f"{c.stage}/{c.tag}: {c.error}" for c in self.calls if c.error],
            "repairs": sum(1 for c in self.calls if c.repaired),
            "wall_seconds_live": round(sum(c.seconds for c in live), 1),
            "per_stage": {st: {"calls": sum(1 for c in self.calls if c.stage == st),
                               "live": sum(1 for c in live if c.stage == st),
                               "live_seconds": round(sum(c.seconds for c in live if c.stage == st), 1),
                               "median_live_seconds": round(sorted(c.seconds for c in live if c.stage == st)[len([c for c in live if c.stage == st]) // 2], 1) if any(c.stage == st for c in live) else 0.0}
                          for st in sorted({c.stage for c in self.calls})},
        }


class RateLimiter:
    def __init__(self, rpm: float):
        self.interval = 60.0 / max(rpm, 0.01)
        self.lock = threading.Lock()
        self.next_ok = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            if now < self.next_ok:
                time.sleep(self.next_ok - now)
            self.next_ok = max(now, self.next_ok) + self.interval


class LLM(ABC):
    provider: str
    model: str

    def __init__(self, provider: str, model: str, cfg: dict, cache_dir: Path | None, log: RunLog):
        self.provider, self.model = provider, model
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.log = log
        self.limiter = RateLimiter(cfg.get("rpm", 8))
        self.max_retries = int(cfg.get("max_retries", 6))
        self.temperature = float(cfg.get("temperature", 0.0))

    @property
    def name(self) -> str:
        return f"{self.provider}/{self.model}"

    # ---- backend contract -------------------------------------------------
    @abstractmethod
    def _raw_call(self, system: str, user: str, schema: type[BaseModel]) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens). Raise for transport errors."""

    @abstractmethod
    def list_models(self) -> list[str]: ...

    # ---- public ------------------------------------------------------------
    def complete_json(self, system: str, user: str, schema: type[T], stage: str, tag: str, raw_path: Path | None = None) -> T:
        key = hashlib.sha256(f"{self.name}\n{system}\n{user}\n{schema.__name__}".encode()).hexdigest()[:24]
        rec = CallRecord(stage=stage, tag=tag, provider=self.provider, model=self.model, prompt_sha=key, cached=False)
        cache_file = self.cache_dir / f"{self.provider}__{self.model.replace('/', '_')}" / f"{key}.json" if self.cache_dir else None
        if cache_file and cache_file.exists():
            data = json.loads(cache_file.read_text())
            try:
                obj = schema.model_validate(_extract_json(data["text"]))
                rec.cached, rec.input_tokens, rec.output_tokens = True, data.get("in", 0), data.get("out", 0)
                rec.response_text = data["text"]
                self.log.add(rec)
                _dump_raw(raw_path, rec)
                return obj
            except (ValidationError, ValueError):
                pass  # stale cache; re-run
        t0 = time.monotonic()
        text, obj, err = "", None, None
        for attempt in range(1, 3):
            prompt_user = user if attempt == 1 else f"{user}\n\nYour previous answer failed validation:\n{err}\nReturn corrected JSON only."
            text, i_tok, o_tok = self._call_with_retries(system, prompt_user, schema)
            rec.input_tokens += i_tok
            rec.output_tokens += o_tok
            rec.attempts = attempt
            try:
                obj = schema.model_validate(_extract_json(text))
                break
            except (ValidationError, ValueError) as e:
                err = str(e)[:1500]
                rec.repaired = True
        rec.seconds = time.monotonic() - t0
        rec.response_text = text
        if obj is None:
            rec.error = f"invalid JSON after repair: {err[:200] if err else ''}"
            self.log.add(rec)
            raise LLMError(rec.error)
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"text": text, "in": rec.input_tokens, "out": rec.output_tokens}))
        self.log.add(rec)
        _dump_raw(raw_path, rec)
        return obj

    def _call_with_retries(self, system: str, user: str, schema: type[BaseModel]) -> tuple[str, int, int]:
        delay = 2.0
        last: Exception | None = None
        status = None
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            try:
                return self._raw_call(system, user, schema)
            except LLMError:
                raise
            except Exception as e:  # noqa: BLE001 - provider SDKs raise many types
                last = e
                status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
                if status == 402 and "in-flight" not in _error_message(e) and hasattr(self, "_rotate_key") and self._rotate_key(_error_message(e)):  # type: ignore[attr-defined]
                    continue
                # 402 from OpenRouter can mean "credit reservation exceeded by in-flight requests": transient, so retry it
                retryable = status in (408, 409, 429, 500, 502, 503, 504, 529) or status is None or (status == 402 and "in-flight" in _error_message(e))
                if not retryable or attempt == self.max_retries:
                    break
                if os.environ.get("PROMO_VERBOSE"):
                    print(f"[retry {attempt + 1}/{self.max_retries}] {self.name}: HTTP {status}: {_error_message(e)[:160]}", file=sys.stderr, flush=True)
                retry_after = None
                headers = getattr(getattr(e, "response", None), "headers", None)
                if headers and headers.get("retry-after"):
                    try:
                        retry_after = float(headers.get("retry-after"))
                    except ValueError:
                        retry_after = None
                time.sleep(min(retry_after or delay, 90.0))
                delay = min(delay * 2, 60.0)
        msg = _error_message(last)
        fatal = status in (400, 401, 403, 404) or (status == 402 and "in-flight" not in msg)  # out of credit: stop, don't degrade silently
        rec = CallRecord(stage="transport", tag=schema.__name__, provider=self.provider, model=self.model, prompt_sha="", cached=False, error=f"{status}: {msg[:200]}")
        self.log.add(rec)
        raise LLMError(f"{self.name}: HTTP {status}: {msg[:400]}" if status else f"{self.name}: {type(last).__name__}: {msg[:400]}", fatal=fatal)


def _error_message(e: Exception | None) -> str:
    """Pull the human-readable message out of provider error bodies when there is one."""
    if e is None:
        return "unknown error"
    body = getattr(e, "body", None)
    try:
        if isinstance(body, list) and body and isinstance(body[0], dict):
            body = body[0]
        if isinstance(body, dict):
            err = body.get("error", body)
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])
            if isinstance(err, str):
                return err
    except Exception:  # noqa: BLE001
        pass
    return str(e)


def _dump_raw(path: Path | None, rec: CallRecord) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "stage": rec.stage, "tag": rec.tag, "provider": rec.provider, "model": rec.model,
        "prompt_sha": rec.prompt_sha, "cached": rec.cached, "attempts": rec.attempts, "repaired": rec.repaired,
        "input_tokens": rec.input_tokens, "output_tokens": rec.output_tokens, "response_text": rec.response_text,
    }, indent=1))


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise ValueError("no JSON object in response")


def parse_model_string(s: str | None) -> tuple[str, str] | None:
    s = (s or os.environ.get("PROMO_MODEL") or "none").strip()
    if s.lower() in ("", "none", "off", "heuristic"):
        return None
    if "/" not in s:
        raise ValueError(f"model must be provider/model, got {s!r}")
    provider, model = s.split("/", 1)
    return provider.lower(), model


def make_llm(model_string: str | None, cfg: dict, cache_dir: Path | None, log: RunLog) -> LLM | None:
    parsed = parse_model_string(model_string)
    if parsed is None:
        return None
    provider, model = parsed
    if provider == "mock":
        from .mock import MockLLM

        return MockLLM(provider, model, {**cfg, "rpm": 100000}, cache_dir, log)
    if provider == "anthropic":
        from .anthropic_client import AnthropicLLM

        return AnthropicLLM(provider, model, cfg, cache_dir, log)
    from .openai_compat import OpenAICompatLLM

    return OpenAICompatLLM(provider, model, cfg, cache_dir, log)
