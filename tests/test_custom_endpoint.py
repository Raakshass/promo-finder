"""'We may connect stronger models through APIs we provide': any OpenAI-compatible endpoint works via custom/<model>."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import BaseModel

from promo_finder.llm import LLMError, RunLog, make_llm


class Out(BaseModel):
    ok: bool
    word: str


class Handler(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "model": body.get("model")})
        resp = {"id": "x", "object": "chat.completion", "created": 0, "model": body.get("model"),
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"ok": true, "word": "hello"}'}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_custom_openai_compatible_endpoint(server, monkeypatch):
    monkeypatch.setenv("PROMO_BASE_URL", server)
    monkeypatch.setenv("PROMO_API_KEY", "kuku-test-key")
    llm = make_llm("custom/kuku-strong-model", {"rpm": 100000, "max_retries": 0}, None, RunLog())
    out = llm.complete_json("system", "user", Out, "test", "t1")
    assert out.ok and out.word == "hello"
    assert Handler.seen[-1]["path"] == "/v1/chat/completions"
    assert Handler.seen[-1]["model"] == "kuku-strong-model"
    assert Handler.seen[-1]["auth"] == "Bearer kuku-test-key"


def test_custom_without_base_url_says_what_to_set(monkeypatch):
    monkeypatch.delenv("PROMO_BASE_URL", raising=False)
    monkeypatch.setenv("PROMO_API_KEY", "k")
    with pytest.raises(LLMError, match="PROMO_BASE_URL"):
        make_llm("custom/m", {"rpm": 100000, "max_retries": 0}, None, RunLog())


class QuotaHandler(BaseHTTPRequestHandler):
    """Answers like OpenRouter when a key's free daily quota is used up (or a per-minute limit, if asked)."""
    calls: list = []
    per_minute = False

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        QuotaHandler.calls.append(self.headers.get("Authorization"))
        kind = "free-models-per-min" if QuotaHandler.per_minute else "free-models-per-day"
        body = json.dumps({"error": {"message": f"Rate limit exceeded: {kind}. Add 10 credits to unlock 1000 free model requests per day", "code": 429}}).encode()
        self.send_response(429)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("retry-after", "0")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture()
def quota_server():
    srv = HTTPServer(("127.0.0.1", 0), QuotaHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_daily_quota_on_every_key_stops_at_once_with_a_clear_message(quota_server, monkeypatch):
    """Honest failure: no minutes of retries, no silent fallback to the keyword twin."""
    QuotaHandler.calls.clear(); QuotaHandler.per_minute = False
    monkeypatch.setenv("PROMO_BASE_URL", quota_server)
    monkeypatch.setenv("PROMO_API_KEY", "key-one")
    monkeypatch.setenv("PROMO_API_KEY2", "key-two")
    llm = make_llm("custom/m", {"rpm": 100000, "max_retries": 3}, None, RunLog())
    with pytest.raises(LLMError) as err:
        llm.complete_json("s", "u", Out, "t", "x")
    assert err.value.fatal and "daily quota" in str(err.value)
    assert QuotaHandler.calls == ["Bearer key-one", "Bearer key-two"]  # rotated once, then stopped: no retry storm


def test_per_minute_limit_is_retried_not_fatal(quota_server, monkeypatch):
    QuotaHandler.calls.clear(); QuotaHandler.per_minute = True
    monkeypatch.setenv("PROMO_BASE_URL", quota_server)
    monkeypatch.setenv("PROMO_API_KEY", "key-one")
    monkeypatch.delenv("PROMO_API_KEY2", raising=False)
    llm = make_llm("custom/m", {"rpm": 100000, "max_retries": 2}, None, RunLog())
    with pytest.raises(LLMError) as err:
        llm.complete_json("s", "u", Out, "t", "x")
    assert not err.value.fatal and len(QuotaHandler.calls) == 3  # first try + 2 retries, same key
    QuotaHandler.per_minute = False
