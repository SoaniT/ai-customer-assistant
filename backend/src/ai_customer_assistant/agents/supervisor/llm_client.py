from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional, Protocol

from groq import Groq
from groq import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    RateLimitError,
    APIStatusError,
)

from .schema import ConversationTurn

# Maximum number of conversation turns the classifier sends to the model.
# gpt-oss-120b on Groq degrades into unparseable JSON output as the
# conversation grows (observed ~50% failure after 3-4 turns); bounding the
# history keeps the prompt small and the classification stable.
_MAX_HISTORY_TURNS: int = 6

# Bounded retries for Groq's intermittent `output_parse_failed` 400: the
# same request frequently succeeds on the next attempt (the failure is
# stochastic, not prompt-dependent), so a couple of retries turn a ~50%
# failure rate into a ~few-percent one.
_MAX_PARSE_RETRIES: int = 3

# Cap the model's output length. gpt-oss-120b on Groq streams as long as it
# wants; unbounded answers were taking 20-40s on later conversation turns
# (team-members listings etc.), blowing past the frontend's timeout. Bounding
# generation keeps worst-case latency predictable.
_MAX_OUTPUT_TOKENS: int = 1024

# Rate-limit retry bound. 429s carry a `retry-after`; only short ones (burst
# limits, e.g. TPM/RPM) are worth waiting for. A long retry-after means a
# daily token budget is exhausted — retrying would burn budget for nothing,
# so raise and let the caller degrade gracefully instead.
_MAX_RATE_LIMIT_WAIT_S: float = 5.0


class SupervisorLLMClient(Protocol):
    """Anything with this shape can back the Supervisor's classification step."""

    def classify(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: list[ConversationTurn],
    ) -> str:
        ...


class StubSupervisorLLMClient:
    """Deterministic stand-in used until a real provider is configured."""

    def classify(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: list[ConversationTurn],
    ) -> str:
        return json.dumps(
            {
                "request_category": "DOMAIN_REQUEST",
                "domain_confidence": 0.0,
                "intent": "UNKNOWN",
                "intent_confidence": 0.0,
                "clarification_question": None,
            }
        )


def _to_gemini_role(role: str) -> str:
    return "model" if role == "assistant" else "user"


def _extract_gemini_text(data: dict) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts)
    except (KeyError, IndexError, TypeError):
        return "{}"


_GEMINI_STATUS_MESSAGES = {
    400: (
        "Gemini rejected the request (400 Bad Request) — usually a malformed "
        "payload or an unsupported model name."
    ),
    403: (
        "Gemini rejected the API key (403 Forbidden)."
    ),
    404: (
        "Gemini returned 404 Not Found — the model name is likely wrong."
    ),
    429: (
        "Gemini's free-tier rate limit was hit (429 Too Many Requests)."
    ),
}


def _gemini_http_error(response, original: Exception) -> Exception:
    message = _GEMINI_STATUS_MESSAGES.get(response.status_code)
    return RuntimeError(message) if message else original


class GeminiSupervisorLLMClient:
    _ENDPOINT_TEMPLATE = (
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        timeout: float = 15.0,
    ) -> None:
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("GEMINI_API_KEY not set.")

        self.api_key = resolved_key
        self.model = model
        self.timeout = timeout

    def classify(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: list[ConversationTurn],
    ) -> str:
        import requests

        contents = [
            *(
                {
                    "role": _to_gemini_role(turn.role),
                    "parts": [{"text": turn.content}],
                }
                for turn in conversation_history
            ),
            {
                "role": "user",
                "parts": [{"text": user_message}],
            },
        ]

        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": contents,
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0,
            },
        }

        response = requests.post(
            self._ENDPOINT_TEMPLATE.format(model=self.model),
            params={"key": self.api_key},
            json=payload,
            timeout=self.timeout,
        )

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise _gemini_http_error(response, exc) from exc

        return _extract_gemini_text(response.json())


class GroqSupervisorLLMClient:
    """Groq-backed implementation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-20b",
        timeout: float = 15.0,
    ) -> None:
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError("GROQ_API_KEY not set.")

        self.client = Groq(api_key=resolved_key)
        self.model = model
        self.timeout = timeout

    def classify(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: list[ConversationTurn],
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            *(
                {
                    "role": turn.role,
                    "content": turn.content,
                }
                for turn in conversation_history[-_MAX_HISTORY_TURNS:]
            ),
            {
                "role": "user",
                "content": user_message,
            },
        ]

        response = self._complete(messages)
        return response.choices[0].message.content or "{}"

    def _complete(self, messages: list[dict]):
        """Call Groq without a ``response_format``, retrying the
        intermittent ``output_parse_failed`` 400 a bounded number of times.

        Deliberately NO ``response_format``: gpt-oss-120b on Groq
        frequently emits output that Groq's JSON-mode parser rejects
        (``output_parse_failed`` / ``json_validate_failed``) once the
        Supervisor prompt + conversation history grow beyond a couple of
        turns. In free-form mode the model instead produces clean, parseable
        JSON matching the prompt's OUTPUT FORMAT block. ``parse_llm_response``
        degrades any residual malformed output to the safe fallback.
        """
        last_error: Optional[Exception] = None
        for attempt in range(_MAX_PARSE_RETRIES):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=_MAX_OUTPUT_TOKENS,
                    reasoning_effort="low",
                )
            except (BadRequestError, APIConnectionError, APITimeoutError, RateLimitError, APIStatusError) as exc:
                last_error = exc
                if not _is_retryable(exc) or attempt == _MAX_PARSE_RETRIES - 1:
                    raise
                time.sleep(_sleep_before_retry(exc, attempt))
        raise last_error


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    """The ``Retry-After`` header Groq sets on 429s (seconds), or None."""
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", None)
    raw = header.get("retry-after") if header is not None else None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _sleep_before_retry(exc: Exception, attempt: int) -> float:
    """Backoff for a retry: honor a short ``Retry-After`` on rate limits,
    otherwise the standard per-attempt backoff."""
    if isinstance(exc, RateLimitError):
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None and retry_after <= _MAX_RATE_LIMIT_WAIT_S:
            return max(retry_after, 0.5 * (attempt + 1))
    return 0.5 * (attempt + 1)


def _is_retryable(exc: Exception) -> bool:
    """True for Groq errors worth retrying: the intermittent
    ``output_parse_failed`` 400, transient transport errors (connection
    blips / timeouts), and short-wait rate limits. A long ``retry-after``
    means the daily token budget is exhausted — re-raise immediately so the
    caller can degrade gracefully. Anything else (auth, bad request for
    another reason) is re-raised immediately."""
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, RateLimitError):
        retry_after = _retry_after_seconds(exc)
        return retry_after is None or retry_after <= _MAX_RATE_LIMIT_WAIT_S
    if isinstance(exc, BadRequestError):
        return "output_parse_failed" in str(getattr(exc, "body", None) or "")
    return False


_PROVIDER_FACTORIES: dict[str, Callable[[], SupervisorLLMClient]] = {
    "stub": StubSupervisorLLMClient,
    "gemini": GeminiSupervisorLLMClient,
    "groq": GroqSupervisorLLMClient,
}


def build_llm_client(provider: Optional[str] = None) -> SupervisorLLMClient:
    resolved_provider = (
        provider
        if provider is not None
        else "groq" if os.environ.get("GROQ_API_KEY")
        else "stub"
    )

    try:
        factory = _PROVIDER_FACTORIES[resolved_provider]
    except KeyError as exc:
        available = sorted(_PROVIDER_FACTORIES)
        raise ValueError(
            f"Unknown provider {resolved_provider!r}. Available: {available}"
        ) from exc

    return factory()