"""One adapter for every OpenAI-compatible endpoint: Gemini, Groq, OpenRouter, OpenAI, Ollama."""
from __future__ import annotations

import os

from openai import BadRequestError, OpenAI, PermissionDeniedError, RateLimitError
from pydantic import BaseModel

from .base import LLM, LLMError

PROVIDERS: dict[str, tuple[str | None, list[str]]] = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", ["GEMINI_API_KEY", "GOOGLE_API_KEY"]),
    "groq": ("https://api.groq.com/openai/v1", ["GROQ_API_KEY"]),
    "openrouter": ("https://openrouter.ai/api/v1", ["OPENROUTER_API_KEY"]),
    "openai": (None, ["OPENAI_API_KEY"]),
    "ollama": ("http://localhost:11434/v1", ["OLLAMA_API_KEY"]),
    "aimlapi": ("https://api.aimlapi.com/v1", ["AIMLAPI_API_KEY"]),
    # free, open-weights models with no daily cap (rate-limited, ~40 requests/min): NVIDIA's API catalog (build.nvidia.com),
    # e.g. nvidia/nvidia/nemotron-3.5-super-vl-120b-a12b, the same model the official run used through OpenRouter
    "nvidia": ("https://integrate.api.nvidia.com/v1", ["NVIDIA_API_KEY"]),
    # Mistral's free Experiment tier (about 1B tokens a month), e.g. mistral/mistral-small-latest
    "mistral": ("https://api.mistral.ai/v1", ["MISTRAL_API_KEY"]),
    # any OpenAI-compatible endpoint (an internal gateway, LiteLLM, vLLM, Together, Fireworks, Azure's /openai/v1, ...):
    # PROMO_MODEL=custom/<model id> with PROMO_BASE_URL and PROMO_API_KEY
    "custom": (None, ["PROMO_API_KEY"]),
}


class OpenAICompatLLM(LLM):
    def __init__(self, provider, model, cfg, cache_dir, log):
        super().__init__(provider, model, cfg, cache_dir, log)
        if provider not in PROVIDERS:
            raise LLMError(f"unknown provider {provider!r}; known: {', '.join(PROVIDERS)} or anthropic")
        base_url, key_names = PROVIDERS[provider]
        # key pool: NAME, NAME2, NAME_2, NAME3 ... so request-capped free tiers can rotate across accounts
        self.keys: list[str] = []
        for k in key_names:
            for suffix in ("", "2", "_2", "3", "_3", "4", "_4", "5", "_5"):
                v = os.environ.get(k + suffix)
                if v and v not in self.keys:
                    self.keys.append(v)
        off = int(os.environ.get("PROMO_KEY_START", "0") or 0)  # parallel runs: start each on a different key of the pool
        if self.keys and off:
            off %= len(self.keys)
            self.keys = self.keys[off:] + self.keys[:off]
        if not self.keys and provider == "ollama":
            self.keys = ["ollama"]
        if not self.keys:
            raise LLMError(f"no API key for {provider}: set {' or '.join(key_names)} in .env")
        if provider == "custom":
            base_url = os.environ.get("PROMO_BASE_URL", "").strip() or None
            if not base_url:
                raise LLMError("custom provider: set PROMO_BASE_URL (the OpenAI-compatible endpoint, e.g. https://gateway.example.com/v1) and PROMO_API_KEY")
        self.base_url = base_url
        self.key_idx = 0
        self.client = self._client()
        self._json_schema_ok = True
        self._json_object_ok = True   # third fallback: no response_format at all (prompts already say "JSON only")
        self._temperature_ok = True
        self._template_kwargs_ok = True  # NVIDIA's API: chat_template_kwargs to switch Nemotron's thinking off
        self.extra_headers = {"HTTP-Referer": "https://github.com/promo-finder", "X-Title": "promo-finder"} if provider == "openrouter" else None

    def _client(self) -> OpenAI:
        return OpenAI(api_key=self.keys[self.key_idx], base_url=self.base_url, timeout=180.0, max_retries=0)

    def _rotate_key(self, reason: str) -> bool:
        """Move to the next key in the pool (daily cap or credit exhausted). Returns False when none are left."""
        if self.key_idx + 1 >= len(self.keys):
            return False
        self.key_idx += 1
        self.client = self._client()
        self.log.add(__import__("promo_finder.llm.base", fromlist=["CallRecord"]).CallRecord(
            stage="transport", tag="key-rotation", provider=self.provider, model=self.model, prompt_sha="", cached=False,
            error=f"rotated to key #{self.key_idx + 1} of {len(self.keys)}: {reason[:120]}"))
        return True

    def _raw_call(self, system, user, schema: type[BaseModel]):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kwargs: dict = {"model": self.model, "messages": messages, "max_tokens": int(self.cfg.get("max_tokens", 4096))}
        if self._temperature_ok:
            kwargs["temperature"] = self.temperature
        if self._json_schema_ok:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": _loosen(schema.model_json_schema())},
            }
        elif self._json_object_ok:
            kwargs["response_format"] = {"type": "json_object"}
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        if self.cfg.get("reasoning_off") and self.provider == "openrouter":
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}
        if self.cfg.get("reasoning_off") and self.provider == "nvidia" and self._template_kwargs_ok:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}  # Nemotron: answer directly
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except (RateLimitError, PermissionDeniedError) as e:
            msg = str(getattr(e, "body", e)).lower()
            # a DAILY cap (free tier used up) is not a per-minute limit: rotate keys, and when none are left stop the run
            # at once with a clear message instead of retrying for minutes and quietly degrading to the keyword twin
            daily = any(w in msg for w in ("per-day", "per day", "daily", "quota")) and not any(w in msg for w in ("per-min", "per minute"))
            if daily:
                if self._rotate_key(msg):
                    return self._raw_call(system, user, schema)
                raise LLMError(f"{self.name}: every key has used its free daily quota ({msg[:140]}). It resets at 00:00 UTC (05:30 IST). "
                               "Use another key or provider, or --model none.", fatal=True)
            raise
        except BadRequestError as e:
            msg = str(e).lower()
            fmt_problem = "schema" in msg or "response_format" in msg or "unsupported" in msg or "json" in msg
            if self._json_schema_ok and fmt_problem:
                self._json_schema_ok = False
                return self._raw_call(system, user, schema)
            if self._json_object_ok and fmt_problem:
                self._json_object_ok = False
                return self._raw_call(system, user, schema)
            if self._temperature_ok and "temperature" in msg:
                self._temperature_ok = False
                return self._raw_call(system, user, schema)
            if self._template_kwargs_ok and ("chat_template_kwargs" in msg or "enable_thinking" in msg or "extra" in msg):
                self._template_kwargs_ok = False
                return self._raw_call(system, user, schema)
            raise
        if not getattr(resp, "choices", None):  # some gateways return 200 with an error body and no choices
            err = getattr(resp, "error", None) or getattr(resp, "model_extra", None) or "empty response"
            e = RuntimeError(f"empty response from provider: {str(err)[:200]}")
            e.status_code = 503  # treat as transient
            raise e
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return text, (usage.prompt_tokens if usage else 0), (usage.completion_tokens if usage else 0)

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list())


def _loosen(schema: dict) -> dict:
    """Strip keywords some providers reject (minimum/maximum/format) and resolve $defs inline."""
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"].split("/")[-1]
                return walk(dict(defs.get(ref, {})))
            return {k: walk(v) for k, v in node.items() if k not in ("minimum", "maximum", "format", "title", "default")}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)
