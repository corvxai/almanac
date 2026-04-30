"""OpenAI provider — mock implementation.

OpenAI's Responses API (/v1/responses) supports a web_search tool that
returns structured output items. The response includes output[] blocks
with type "web_search_call" containing search queries and results.
Text output blocks include annotations[] with URLs.

The older Chat Completions API (/v1/chat/completions) does not provide
source metadata — just the completion text.

Visibility tier: HIGH with Responses API + web_search tool.
LOW with plain Chat Completions (text only).
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

from src.core.schemas import ProviderTier
from src.gateway.providers.base import BaseProvider

_MODELS = ["gpt-5.2", "gpt-5", "gpt-5-mini", "o3", "o4-mini"]

_MOCK_ANNOTATIONS = [
    {"type": "url_citation", "url": "https://reuters.com/fed-rate-outlook", "title": "Fed Rate Outlook"},
    {"type": "url_citation", "url": "https://polymarket.com/event/fed-rate", "title": "Polymarket: Fed Rate"},
    {"type": "url_citation", "url": "https://nytimes.com/economy/inflation-update", "title": "NYT: Inflation Update"},
    {"type": "url_citation", "url": "https://wsj.com/economy/fed-policy", "title": "WSJ: Fed Policy"},
    {"type": "url_citation", "url": "https://bbc.co.uk/news/business-economy", "title": "BBC: Business Economy"},
]


class OpenAIProvider(BaseProvider):
    provider_id: str = "openai"
    provider_tier: ProviderTier = ProviderTier.INFERENCE

    def call(self, call_type: str, params: dict[str, Any]) -> dict[str, Any]:
        handlers = {
            "responses": self._mock_responses,
            "chat_completion": self._mock_chat_completion,
        }
        handler = handlers.get(call_type)
        if handler is None:
            raise ValueError(f"Unknown call_type for openai: {call_type}")
        return handler(params)

    def summarise_params(self, call_type: str, params: dict[str, Any]) -> str:
        model = params.get("model", _MODELS[0])
        tools = [t.get("type", "?") for t in params.get("tools", [])]
        tool_str = f", tools=[{','.join(tools)}]" if tools else ""
        return f"openai.{call_type}(model={model}{tool_str})"

    def _mock_responses(self, params: dict[str, Any]) -> dict[str, Any]:
        """Mock the Responses API (/v1/responses) with optional web_search."""
        model = params.get("model", _MODELS[0])
        tools = params.get("tools", [])
        has_web_search = any(t.get("type") == "web_search" for t in tools)

        prediction = round(random.uniform(0.15, 0.85), 4)
        reasoning = (
            f"Analyzing available data and market signals, "
            f"I estimate a {prediction:.1%} probability."
        )

        output_items: list[dict[str, Any]] = []

        if has_web_search:
            output_items.append({
                "type": "web_search_call",
                "id": f"ws_{random.randint(1000, 9999)}",
                "status": "completed",
            })

        num_annotations = random.randint(2, 4) if has_web_search else 0
        annotations = random.sample(_MOCK_ANNOTATIONS, num_annotations) if num_annotations else []

        output_items.append({
            "type": "message",
            "id": f"msg_{random.randint(100000, 999999)}",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": f"PREDICTION: {prediction}\nREASONING: {reasoning}",
                "annotations": annotations,
            }],
        })

        return {
            "id": f"resp_{random.randint(100000, 999999)}",
            "object": "response",
            "model": model,
            "created_at": int(datetime.now(timezone.utc).timestamp()),
            "output": output_items,
            "usage": {
                "input_tokens": random.randint(200, 600),
                "output_tokens": random.randint(150, 400),
                "total_tokens": random.randint(400, 1000),
            },
        }

    def _mock_chat_completion(self, params: dict[str, Any]) -> dict[str, Any]:
        """Mock the Chat Completions API — no source metadata."""
        model = params.get("model", _MODELS[0])
        prediction = round(random.uniform(0.15, 0.85), 4)
        reasoning = f"Based on general analysis, probability is approximately {prediction:.1%}."

        return {
            "id": f"chatcmpl-{random.randint(100000, 999999)}",
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
                "prompt_tokens": random.randint(200, 500),
                "completion_tokens": random.randint(100, 300),
                "total_tokens": random.randint(400, 800),
            },
        }
