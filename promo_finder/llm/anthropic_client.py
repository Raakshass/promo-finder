"""Native Anthropic backend. Structured outputs via messages.parse(); refusal stop reasons surface as errors."""
from __future__ import annotations

import anthropic
from pydantic import BaseModel

from .base import LLM, LLMError


class AnthropicLLM(LLM):
    def __init__(self, provider, model, cfg, cache_dir, log):
        super().__init__(provider, model, cfg, cache_dir, log)
        self.client = anthropic.Anthropic(max_retries=0, timeout=180.0)
        self._parse_ok = True

    def _raw_call(self, system, user, schema: type[BaseModel]):
        if self._parse_ok:
            try:
                resp = self.client.messages.parse(
                    model=self.model,
                    max_tokens=8000,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    output_format=schema,
                )
                if getattr(resp, "stop_reason", None) == "refusal":
                    raise LLMError("model refused the request")
                text = resp.parsed_output.model_dump_json() if resp.parsed_output is not None else _text_of(resp)
                return text, resp.usage.input_tokens, resp.usage.output_tokens
            except anthropic.BadRequestError as e:
                if "output_format" in str(e) or "schema" in str(e).lower():
                    self._parse_ok = False  # older model or unsupported schema; fall back to plain JSON
                else:
                    raise
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=8000,
            system=system + "\nReturn only a JSON object, no prose.",
            messages=[{"role": "user", "content": user}],
        )
        if getattr(resp, "stop_reason", None) == "refusal":
            raise LLMError("model refused the request")
        return _text_of(resp), resp.usage.input_tokens, resp.usage.output_tokens

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list())


def _text_of(resp) -> str:
    return "".join(getattr(b, "text", "") for b in resp.content)
