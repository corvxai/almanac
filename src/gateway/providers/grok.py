"""Grok (xAI) provider — mock implementation.

Grok's API is OpenAI-compatible. When search is enabled (search_parameters.mode = "on"),
the response includes a top-level `citations` array of URLs and
usage.num_sources_used count. The model can also return reasoning_content
alongside the main content.

Visibility tier: HIGH when search is on — we get citations[] with URLs.
Standard completion without search: text only, no source metadata.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

from src.core.schemas import ProviderTier
from src.gateway.providers.base import BaseProvider

_MOCK_CITATIONS = [
    "https://polymarket.com/event/fed-rate-cut",
    "https://reuters.com/markets/us-fed-outlook",
    "https://en.wikipedia.org/wiki/Federal_funds_rate",
    "https://x.com/zerohedge/status/123456789",
    "https://bloomberg.com/news/fed-rate-decision",
]

_MODELS = ["grok-4-0709", "grok-3", "grok-3-mini"]


class GrokProvider(BaseProvider):
    provider_id: str = "grok"
    provider_tier: ProviderTier = ProviderTier.INFERENCE

    def call(self, call_type: str, params: dict[str, Any]) -> dict[str, Any]:
        if call_type != "chat_completion":
            raise ValueError(f"Unknown call_type for grok: {call_type}")
        return self._mock_chat_completion(params)

    def summarise_params(self, call_type: str, params: dict[str, Any]) -> str:
        model = params.get("model", _MODELS[0])
        search = params.get("search", True)
        return f'grok.{call_type}(model={model}, search={"on" if search else "off"})'

    def _mock_chat_completion(self, params: dict[str, Any]) -> dict[str, Any]:
        model = params.get("model", _MODELS[0])
        search_enabled = params.get("search", True)
        prompt_preview = str(params.get("messages", [{}])[-1].get("content", ""))[:100]

        prediction = round(random.uniform(0.15, 0.85), 4)
        reasoning = (
            f"Based on current market data and recent developments, "
            f"I estimate a {prediction:.1%} probability. "
            f"Key factors include recent policy signals and market positioning."
        )

        response: dict[str, Any] = {
            "id": f"chatcmpl-{random.randint(100000, 999999)}",
            "object": "chat.completion",
            "model": model,
            "created": int(datetime.now(timezone.utc).timestamp()),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"PREDICTION: {prediction}\nREASONING: {reasoning}",
                    "reasoning_content": (
                        f"Let me analyze this step by step. The prompt asks about: "
                        f"{prompt_preview}..."
                    ) if "grok-3" not in model else None,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": random.randint(200, 500),
                "completion_tokens": random.randint(100, 400),
                "total_tokens": random.randint(400, 900),
                "num_sources_used": 0,
            },
        }

        if search_enabled:
            num_citations = random.randint(2, 5)
            selected = random.sample(_MOCK_CITATIONS, num_citations)
            response["citations"] = selected
            response["usage"]["num_sources_used"] = num_citations

        return response
