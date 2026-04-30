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
from typing import Any

import httpx

from src.core.schemas import ProviderTier
from src.gateway.providers.base import BaseProvider

API_URL = "https://api.anthropic.com/v1/messages"
# Anthropic requires this literal for the Messages API (see API reference).
API_VERSION = "2023-06-01"


class ClaudeProvider(BaseProvider):
    provider_id: str = "claude"
    provider_tier: ProviderTier = ProviderTier.INFERENCE

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def call(self, call_type: str, params: dict[str, Any]) -> dict[str, Any]:
        if call_type != "messages":
            raise ValueError(f"Unknown call_type for claude: {call_type}")
        return self._send_messages(params)

    def summarise_params(self, call_type: str, params: dict[str, Any]) -> str:
        model = params.get("model", "unknown")
        tools = [t.get("type", t.get("name", "?")) for t in params.get("tools", [])]
        tool_str = f", tools=[{','.join(tools)}]" if tools else ""
        return f"claude.{call_type}(model={model}{tool_str})"

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
