"""Claude (Anthropic) provider — real implementation using the Messages API.

Claude's Messages API returns structured content blocks. When the web_search
server tool is enabled, the response includes WebSearchToolResultBlock items
containing the actual search results with URLs, titles, and page_age.
Text blocks may also include citations[] with type "web_search_result_location"
linking cited_text to specific URLs.

Visibility tier: HIGH when web_search tool is used — we get full search
results with URLs and structured citations linking text spans to sources.
Without web_search: text only, no source metadata.
"""

from __future__ import annotations

import os
import random
from typing import Any

import httpx

from src.core.schemas import ProviderTier
from src.gateway.providers.base import BaseProvider

API_URL = "https://api.anthropic.com/v1/messages"
# Anthropic requires this literal for the Messages API (see API reference).
API_VERSION = "2023-06-01"


class ClaudeProvider(BaseProvider):
    provider_id: str = "anthropic"
    provider_tier: ProviderTier = ProviderTier.INFERENCE

    def __init__(self, api_key: str | None = None, *, use_mock: bool = False):
        self._api_key = (api_key or os.environ.get("ANTHROPIC_API_KEY", "")).strip()
        # Live only when a key is present and mock wasn't forced. The simulated
        # gateway runs mock-by-default so verifier/miner tests need no key, no
        # network, and no spend (see scripts/run_gateway.py).
        self._live = bool(self._api_key) and not use_mock

    @property
    def mode(self) -> str:
        """``"live"`` when an API key is configured (and not ``use_mock``), else ``"mock"``."""
        return "live" if self._live else "mock"

    def call(self, call_type: str, params: dict[str, Any]) -> dict[str, Any]:
        if call_type != "messages":
            raise ValueError(f"Unknown call_type for claude: {call_type}")
        if not self._live:
            return self._mock_messages(params)
        return self._send_messages(params)

    def summarise_params(self, call_type: str, params: dict[str, Any]) -> str:
        model = params.get("model", "unknown")
        tools = [t.get("type", t.get("name", "?")) for t in params.get("tools", [])]
        tool_str = f", tools=[{','.join(tools)}]" if tools else ""
        return f"claude.{call_type}(model={model}{tool_str})"

    def _mock_messages(self, params: dict[str, Any]) -> dict[str, Any]:
        """Deterministic-shape Anthropic Messages payload — no key, no network.

        Mirrors the real Messages response (content blocks + usage) closely
        enough that the validator's extractor/recorder treats it like a live
        call, so end-to-end sandbox tests run fully offline.
        """
        model = params.get("model", "claude-haiku-4-5-20250414")
        prediction = round(random.uniform(0.15, 0.85), 4)
        reasoning = (
            f"Mock analysis: weighing recent signals and base rates, I estimate a "
            f"{prediction:.1%} probability."
        )
        return {
            "id": f"msg_mock_{random.randint(100000, 999999)}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [
                {"type": "text", "text": f"PREDICTION: {prediction}\nREASONING: {reasoning}"}
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": random.randint(300, 700),
                "output_tokens": random.randint(100, 400),
            },
        }

    def _send_messages(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Provide it via constructor or environment."
            )

        model = params.get("model", "claude-haiku-4-5-20250414")
        messages = params.get("messages", [])
        max_tokens = params.get("max_tokens", 1024)

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        if "system" in params:
            body["system"] = params["system"]

        if "tools" in params:
            body["tools"] = params["tools"]

        if "temperature" in params:
            body["temperature"] = params["temperature"]

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

        with httpx.Client(timeout=90.0) as client:
            resp = client.post(API_URL, json=body, headers=headers)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = resp.text[:2000] if resp.text else "(empty body)"
                raise RuntimeError(
                    f"Anthropic API {resp.status_code}: {detail}",
                ) from exc
            return resp.json()
