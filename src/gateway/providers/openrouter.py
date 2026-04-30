"""OpenRouter provider — OpenAI-compatible chat completions (mock or live).

With ``OPENROUTER_API_KEY`` set, calls ``https://openrouter.ai/api/v1/chat/completions``.
Otherwise returns the same shaped mock payload used in local development.

OpenRouter is a routing layer providing access to multiple LLM providers
through a unified OpenAI-compatible chat completions API. The response
format mirrors standard chat completions — no built-in search or citations.

Visibility tier: LOW — text completion only, no source metadata.
The value is model diversity (access Claude, Gemini, Llama, etc. through
one API), not evidence transparency.
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timezone
from typing import Any

import httpx

from src.core.schemas import ProviderTier
from src.gateway.providers.base import BaseProvider

API_URL = "https://openrouter.ai/api/v1/chat/completions"

_MODELS = [
    "anthropic/claude-sonnet-4-6",
    "google/gemini-2.5-flash",
    "anthropic/claude-haiku-4-5",
    "meta-llama/llama-4-maverick",
    "deepseek/deepseek-r1",
]

# Default when none is passed (live path) — pick a small, common OpenRouter slug.
_DEFAULT_LIVE_MODEL = "openai/gpt-4o-mini"


class OpenRouterProvider(BaseProvider):
    provider_id: str = "openrouter"
    provider_tier: ProviderTier = ProviderTier.INFERENCE

    def __init__(self, api_key: str | None = None, *, use_mock: bool = False):
        key = (api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY", "")).strip()
        self._api_key = key
        if use_mock:
            self._live = False
        else:
            self._live = bool(key)

    @property
    def mode(self) -> str:
        """``\"live\"`` when API key is configured (and not ``use_mock``), else ``\"mock\"``."""
        return "live" if self._live else "mock"

    def call(self, call_type: str, params: dict[str, Any]) -> dict[str, Any]:
        if call_type != "chat_completion":
            raise ValueError(f"Unknown call_type for openrouter: {call_type}")
        if self._live:
            return self._live_chat_completion(params)
        return self._mock_chat_completion(params)

    def summarise_params(self, call_type: str, params: dict[str, Any]) -> str:
        model = params.get("model", _MODELS[0])
        return f"openrouter.{call_type}(model={model})"

    def _live_chat_completion(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Pass api_key=... or set the environment variable."
            )
        model = params.get("model", _DEFAULT_LIVE_MODEL)
        messages = params.get("messages", [])
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if "max_tokens" in params:
            body["max_tokens"] = params["max_tokens"]
        if "temperature" in params:
            body["temperature"] = params["temperature"]

        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        headers["X-Title"] = os.environ.get("OPENROUTER_APP_TITLE", "arcratio-gateway")

        with httpx.Client(timeout=90.0) as client:
            resp = client.post(API_URL, json=body, headers=headers)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = resp.text[:2000] if resp.text else "(empty body)"
                raise RuntimeError(
                    f"OpenRouter API {resp.status_code}: {detail}",
                ) from exc
            return resp.json()

    def _mock_chat_completion(self, params: dict[str, Any]) -> dict[str, Any]:
        model = params.get("model", _MODELS[0])
        prediction = round(random.uniform(0.15, 0.85), 4)

        reasoning = (
            f"Analyzing the event based on historical patterns and current context, "
            f"I estimate a {prediction:.1%} probability. Key considerations include "
            f"recent policy signals, market positioning, and expert consensus."
        )

        return {
            "id": f"gen-{random.randint(100000, 999999)}",
            "object": "chat.completion",
            "model": model,
            "created": int(datetime.now(timezone.utc).timestamp()),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"PREDICTION: {prediction}\nREASONING: {reasoning}",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": random.randint(300, 700),
                "completion_tokens": random.randint(100, 400),
                "total_tokens": random.randint(500, 1100),
            },
        }
