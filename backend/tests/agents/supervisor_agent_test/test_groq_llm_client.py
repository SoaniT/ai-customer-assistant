"""Tests for GroqSupervisorLLMClient's parse-failure robustness.

Covers the two mitigations added for gpt-oss-120b's intermittent
`output_parse_failed` 400s (which surfaced after 2-3 chat turns as
history grew):

  1. the conversation history sent to the classifier is capped at the
     last `_MAX_HISTORY_TURNS` turns;
  2. a bounded retry retries a `output_parse_failed` BadRequestError
     instead of crashing the graph.

All tests run against a fake Groq client — no network.
"""
from __future__ import annotations

import pytest

import agents.supervisor.llm_client as llm_client_mod
from agents.supervisor.llm_client import GroqSupervisorLLMClient
from agents.supervisor.schema import ConversationTurn


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = type("_Msg", (), {"content": content})()


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs["messages"])
        return _FakeResponse('{"intent": "UNKNOWN"}')


class _FakeGroq:
    def __init__(self, *, api_key: str) -> None:
        self.chat = type("_Chat", (), {"completions": _FakeChatCompletions()})()


@pytest.fixture
def make_client(monkeypatch):
    """Build a GroqSupervisorLLMClient whose client is a fake Groq."""

    def _make() -> tuple[GroqSupervisorLLMClient, _FakeGroq]:
        fake = _FakeGroq(api_key="k")
        monkeypatch.setattr(llm_client_mod, "Groq", lambda api_key: fake)
        return GroqSupervisorLLMClient(api_key="test-key"), fake

    return _make


class TestHistoryCap:
    def test_classify_sends_only_last_max_history_turns(self, make_client):
        client, fake = make_client()
        history = [
            ConversationTurn(role="user", content=f"turn-{i}")
            for i in range(20)
        ]
        client.classify("system", "latest", history)

        messages = fake.chat.completions.calls[0]
        # system + capped history + current user message
        assert len(messages) == llm_client_mod._MAX_HISTORY_TURNS + 2
        # Only the most recent turns survive; the oldest ones are dropped.
        assert messages[1]["content"] == "turn-14"
        assert messages[-2]["content"] == "turn-19"
        assert messages[-1]["content"] == "latest"
        assert not any(m["content"] == "turn-0" for m in messages)


class TestRetryOnParseFailure:
    def test_retries_output_parse_failed_then_succeeds(self, make_client, monkeypatch):
        client, fake = make_client()

        class _FakeBadRequestError(Exception):
            body = {"error": {"code": "output_parse_failed"}}

        monkeypatch.setattr(llm_client_mod, "BadRequestError", _FakeBadRequestError)
        original_create = fake.chat.completions.create
        calls = {"n": 0}

        def flaky_create(**kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _FakeBadRequestError("parse failed")
            return _FakeResponse('{"intent": "KNOWLEDGE_QUERY"}')

        fake.chat.completions.create = flaky_create

        raw = client.classify("system", "question", [])
        assert calls["n"] == 3
        assert raw == '{"intent": "KNOWLEDGE_QUERY"}'

    def test_gives_up_after_max_retries(self, make_client, monkeypatch):
        client, fake = make_client()

        class _FakeBadRequestError(Exception):
            body = {"error": {"code": "output_parse_failed"}}

        monkeypatch.setattr(llm_client_mod, "BadRequestError", _FakeBadRequestError)
        attempts = {"n": 0}

        def always_fail(**kwargs):
            attempts["n"] += 1
            raise _FakeBadRequestError("parse failed")

        fake.chat.completions.create = always_fail

        with pytest.raises(_FakeBadRequestError):
            client.classify("system", "question", [])
        assert attempts["n"] == llm_client_mod._MAX_PARSE_RETRIES


class TestRateLimitHandling:
    def _fake_rate_limit(self, retry_after: str):
        class _FakeHeaders:
            def get(self, name, default=None):
                return retry_after if name == "retry-after" else default

        class _FakeResponse:
            headers = _FakeHeaders()

        class _FakeRateLimitError(Exception):
            def __init__(self, message):
                super().__init__(message)
                self.response = _FakeResponse()

        return _FakeRateLimitError

    def test_retries_short_rate_limit_then_succeeds(self, make_client, monkeypatch):
        client, fake = make_client()
        fake_error = self._fake_rate_limit("2")
        monkeypatch.setattr(llm_client_mod, "RateLimitError", fake_error)
        calls = {"n": 0}

        def flaky_create(**kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise fake_error("rate limited")
            return _FakeResponse('{"intent": "KNOWLEDGE_QUERY"}')

        fake.chat.completions.create = flaky_create

        raw = client.classify("system", "question", [])
        assert calls["n"] == 2
        assert raw == '{"intent": "KNOWLEDGE_QUERY"}'

    def test_long_rate_limit_raises_immediately(self, make_client, monkeypatch):
        client, fake = make_client()
        fake_error = self._fake_rate_limit("500")
        monkeypatch.setattr(llm_client_mod, "RateLimitError", fake_error)
        calls = {"n": 0}

        def always_fail(**kwargs):
            calls["n"] += 1
            raise fake_error("daily budget exhausted")

        fake.chat.completions.create = always_fail

        with pytest.raises(fake_error):
            client.classify("system", "question", [])
        # No retry: a long Retry-After means the daily token budget is gone.
        assert calls["n"] == 1

    def test_rate_limit_without_retry_after_is_retried(self, make_client, monkeypatch):
        client, fake = make_client()
        fake_error = self._fake_rate_limit(None)
        monkeypatch.setattr(llm_client_mod, "RateLimitError", fake_error)
        calls = {"n": 0}

        def flaky_create(**kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise fake_error("rate limited")
            return _FakeResponse('{"intent": "KNOWLEDGE_QUERY"}')

        fake.chat.completions.create = flaky_create

        raw = client.classify("system", "question", [])
        assert calls["n"] == 2
        assert raw == '{"intent": "KNOWLEDGE_QUERY"}'
